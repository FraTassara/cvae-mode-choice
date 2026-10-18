#!/usr/bin/env python3
"""
Paso 2 — Generar los conjuntos de entrenamiento de cada partición.

Para cada método y cada partición, produce el conjunto con el que R va a
estimar. El punto importante: **el sobremuestreo se aplica solo a las filas de
entrenamiento de esa partición**, nunca al dataset completo. Si se generan
sintéticos antes de partir en folds, las filas sintéticas del entrenamiento se
construyeron a partir de filas reales que caen en validación.

Salida:

  data/processed/{ds}_{metodo}_train.csv.gz
      Formato largo: una fila por observación de entrenamiento, con columnas
      `repeticion`, `fold` e `is_synthetic` además de los datos.
      Un archivo por método en vez de 50 sueltos.

Uso:
    python python/scripts/02_sinteticos.py --dataset lc_centro
    python python/scripts/02_sinteticos.py --dataset lc_centro --metodo CVAE
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# TensorFlow en modo grafo (v1) emite avisos de "operación modificada tras
# ejecutarse" cada vez que se compila un modelo nuevo — y aquí se entrena uno
# por partición. Son inocuos: el KL annealing se verificó funcionando. Nivel 2
# oculta INFO y WARNING pero DEJA VISIBLES LOS ERRORES reales.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import cargar  # noqa: E402


# --------------------------------------------------------------------------- #
#  Un método de balanceo = una función (train_original) -> sinteticos
#  Devuelve SOLO las filas nuevas; el paso de concatenar es común.
# --------------------------------------------------------------------------- #
def m_original(train, cfg, ds, semilla):
    """Línea base: no genera nada."""
    return train.iloc[0:0].copy()


def m_ros(train, cfg, ds, semilla):
    """Random oversampling con reemplazo hasta igualar la clase mayoritaria."""
    choice = cfg.dataset(ds)["choice_col"]
    objetivo = train[choice].value_counts().max()
    partes = []
    for clase, n in train[choice].value_counts().items():
        faltan = objetivo - n
        if faltan > 0:
            partes.append(train[train[choice] == clase].sample(
                faltan, replace=True, random_state=semilla))
    return pd.concat(partes, ignore_index=True) if partes else train.iloc[0:0].copy()


def m_smote(train, cfg, ds, semilla):
    from imblearn.over_sampling import SMOTE

    choice = cfg.dataset(ds)["choice_col"]
    cols = [c for c in train.columns if c not in ("row_id",)]
    X = train[cols].drop(columns=[choice])
    y = train[choice]
    Xr, yr = SMOTE(random_state=semilla).fit_resample(X, y)
    nuevo = pd.concat([Xr, yr.rename(choice)], axis=1).iloc[len(train):]
    nuevo["row_id"] = -1
    if cfg.dataset(ds).get("acoplar_disponibilidad"):
        nuevo = _acoplar_disponibilidad(nuevo, cfg, ds)
    return nuevo.reset_index(drop=True)


def _ceros_estructurales(cfg, ds) -> dict:
    """
    {columna: (dependientes, valor que las anula)} leído de `cvae.variables`:
    toda variable con `depende` (disponibilidades con anula_si = 0 por
    defecto; p. ej. GA con anula_si = 1). Es la misma información que usa
    VariableSchema.propagate_structural_zeros() en el CVAE.
    """
    variables = (cfg.dataset(ds).get("cvae") or {}).get("variables") or {}
    return {col: (list(spec["depende"]), int(spec.get("anula_si", 0)))
            for col, spec in variables.items()
            if isinstance(spec, dict) and spec.get("depende")}


def _acoplar_disponibilidad(sint: pd.DataFrame, cfg, ds) -> pd.DataFrame:
    """
    SMOTE interpola también las variables que anulan a otras: entre un vecino
    con 0 y otro con 1 sale un valor fraccionario, con atributos interpolados
    > 0. Se redondea al progenitor más cercano (>= 0.5 -> 1) y, si queda en
    el valor que anula, sus dependientes se ponen en 0.
    """
    out = sint.copy()
    for col, (deps, anula_si) in _ceros_estructurales(cfg, ds).items():
        if col not in out:
            continue
        out[col] = (out[col] >= 0.5).astype(int)
        deps = [c for c in deps if c in out]
        out.loc[out[col] == anula_si, deps] = 0
    return out


def _verificar_disponibilidad(sint: pd.DataFrame, cfg, ds, metodo) -> None:
    """
    Comprueba en las filas sintéticas que cada variable con `depende` sea 0/1
    y que, cuando vale `anula_si`, sus dependientes estén en 0. Si el dataset
    declara `acoplar_disponibilidad`, un fallo aborta; si no, solo avisa
    (SM-Centro y LC-Centro: SMOTE interpola DISP*, defecto documentado).
    """
    if sint.empty:
        return
    for col, (deps, anula_si) in _ceros_estructurales(cfg, ds).items():
        if col not in sint:
            continue
        deps = [c for c in deps if c in sint]
        no_binaria = int((~sint[col].isin([0, 1])).sum())
        activa = (sint[col] >= 0.5) if anula_si == 1 else (sint[col] < 0.5)
        sin_acoplar = int((activa & (sint[deps] != 0).any(axis=1)).sum()) if deps else 0
        if no_binaria or sin_acoplar:
            msg = (f"[{ds} / {metodo}] {col}: {no_binaria} filas sintéticas no "
                   f"binarias y {sin_acoplar} con {col} = {anula_si} pero "
                   f"{deps} != 0")
            if cfg.dataset(ds).get("acoplar_disponibilidad"):
                raise SystemExit("ERROR " + msg)
            print("    AVISO " + msg)


def m_class_weights(train, cfg, ds, semilla):
    """
    §2.9 — Baseline con pesos de clase. No genera datos: el desbalance se
    corrige ponderando la verosimilitud con el inverso de la frecuencia de
    clase. El peso lo calcula `_pesos()` y lo consume Apollo vía la columna
    `peso` (apollo_control$weights).
    """
    return train.iloc[0:0].copy()


#: Nivel de elipse que sobrescribe el de config.yaml (§2.7). Lo fija main().
_CONFIANZA: float | None = None


def m_cvae(train, cfg, ds, semilla):
    """
    Entrena un CVAE con las filas de entrenamiento de ESTA partición y genera
    sintéticos. Es el paso caro: un modelo por partición.
    """
    from cvae.pipeline import generar_sinteticos_cvae

    return generar_sinteticos_cvae(train, cfg, ds, semilla, confianza=_CONFIANZA)


METODOS = {
    "original": m_original,
    "ROS": m_ros,
    "SMOTE": m_smote,
    "CVAE": m_cvae,
    "class_weights": m_class_weights,   # §2.9
}


def _pesos(bloque: pd.DataFrame, choice_col: str, metodo: str) -> np.ndarray:
    """
    Columna `peso` que lee Apollo. Vale 1 en todos los métodos salvo
    `class_weights`, donde es el inverso de la frecuencia de clase
    normalizado a media 1 (así el tamaño muestral efectivo no cambia).
    """
    if metodo != "class_weights":
        return np.ones(len(bloque))
    frec = bloque[choice_col].value_counts(normalize=True)
    w = bloque[choice_col].map(lambda c: 1.0 / frec[c]).to_numpy()
    return w / w.mean()


# --------------------------------------------------------------------------- #
def procesar(cfg, ds: str, metodo: str, confianza: float | None = None,
             sufijo: str = "") -> None:
    original = pd.read_csv(cfg.ruta_original(ds))
    folds = pd.read_csv(cfg.ruta_folds(ds))
    fn = METODOS[metodo]

    salida = []
    particiones = folds[folds.split == "train"].groupby(["repeticion", "fold"])
    total = particiones.ngroups
    t0 = time.time()

    for k, ((rep, fold), g) in enumerate(particiones, start=1):
        train = original[original.row_id.isin(g.row_id)].reset_index(drop=True)

        # Semilla derivada de (semilla base, repetición, fold): reproducible
        # y distinta en cada partición.
        semilla = int(cfg.semillas["oversampling"] * 10_000 + rep * 100 + fold)

        sint = fn(train, cfg, ds, semilla)
        _verificar_disponibilidad(sint, cfg, ds, metodo)

        train = train.assign(is_synthetic=0)
        if len(sint):
            sint = sint.assign(is_synthetic=1)
        bloque = pd.concat([train, sint], ignore_index=True)
        bloque["peso"] = _pesos(bloque, cfg.dataset(ds)["choice_col"], metodo)
        bloque.insert(0, "fold", fold)
        bloque.insert(0, "repeticion", rep)
        salida.append(bloque)

        if k % 10 == 0 or k == total:
            print(f"    {k}/{total} particiones  ({time.time() - t0:.0f}s)")

    df = pd.concat(salida, ignore_index=True)
    destino = cfg.ruta_train(ds, metodo + sufijo)
    df.to_csv(destino, index=False, compression="gzip")

    n_sint = int(df.is_synthetic.sum())
    print(f"    {len(df)} filas ({n_sint} sintéticas) -> {destino.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metodo", default=None, help="por defecto, todos los de config")
    ap.add_argument("--confianza", type=float, default=None,
                    help="§2.7: nivel de elipse (0.70/0.90/0.95). Sobrescribe "
                         "config.yaml y guarda con sufijo, p.ej. CVAE_conf90.")
    args = ap.parse_args()

    global _CONFIANZA
    _CONFIANZA = args.confianza
    sufijo = "" if args.confianza is None else f"_conf{int(args.confianza*100)}"
    if args.confianza is not None and args.metodo != "CVAE":
        raise SystemExit("--confianza solo aplica al método CVAE "
                         "(usa --metodo CVAE).")

    cfg = cargar()
    cfg.verificar_manifiesto()

    metodos = [args.metodo] if args.metodo else cfg.metodos
    for m in metodos:
        if m not in METODOS:
            raise SystemExit(f"Método desconocido: {m}. Opciones: {list(METODOS)}")
        print(f"\n[{args.dataset} / {m}{sufijo}]")
        procesar(cfg, args.dataset, m, sufijo=sufijo)


if __name__ == "__main__":
    main()