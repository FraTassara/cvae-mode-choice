#!/usr/bin/env python3
"""
Tests del contrato entre Python y R.

No prueban el modelo: prueban que los archivos que Python le entrega a R
cumplen las invariantes de las que R depende. Si alguna falla, los resultados
serían inválidos aunque todo corriera sin error.

Uso:
    python python/scripts/tests_contrato.py --dataset sm_centro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import cargar  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    cfg = cargar()
    ds = args.dataset
    d = cfg.dataset(ds)
    choice = d["choice_col"]

    original = pd.read_csv(cfg.ruta_original(ds))
    folds = pd.read_csv(cfg.ruta_folds(ds))

    fallos = []

    def chequear(cond, msg):
        print(f"  [{'ok' if cond else 'FALLO'}] {msg}")
        if not cond:
            fallos.append(msg)

    print(f"\n{ds}: estructura")
    chequear(original.row_id.is_unique, "row_id es único en el original")
    chequear(set(original[choice]) <= set(d["clases"]),
             f"solo aparecen las clases declaradas {d['clases']}")

    print(f"\n{ds}: particiones")
    grupos = folds.groupby(["repeticion", "fold"])
    esperadas = cfg.validacion["n_splits"] * cfg.validacion["n_repeats"]
    chequear(grupos.ngroups == esperadas,
             f"hay {esperadas} particiones (encontradas {grupos.ngroups})")

    solapan = []
    incompletas = []
    for (rep, fold), g in grupos:
        tr = set(g.row_id[g.split == "train"])
        va = set(g.row_id[g.split == "valid"])
        if tr & va:
            solapan.append((rep, fold))
        if tr | va != set(original.row_id):
            incompletas.append((rep, fold))
    chequear(not solapan, "train y valid nunca se solapan")
    chequear(not incompletas, "train + valid cubren todas las filas")

    # Cada fila cae exactamente una vez en validación por repetición.
    val = folds[folds.split == "valid"]
    conteo = val.groupby(["repeticion", "row_id"]).size()
    chequear((conteo == 1).all(),
             "cada fila cae en validación exactamente una vez por repetición")

    print(f"\n{ds}: conjuntos de entrenamiento")
    for metodo in cfg.metodos:
        ruta = cfg.ruta_train(ds, metodo)
        if not ruta.exists():
            print(f"  [--] {metodo}: no generado, se salta")
            continue

        train = pd.read_csv(ruta, compression="gzip")

        chequear("is_synthetic" in train.columns,
                 f"{metodo}: existe la columna is_synthetic")

        # LA INVARIANTE CRÍTICA: ninguna fila REAL del entrenamiento puede
        # estar en la validación de su propia partición.
        malas = 0
        for (rep, fold), g in train.groupby(["repeticion", "fold"]):
            reales = set(g.loc[g.is_synthetic == 0, "row_id"])
            va = set(folds.row_id[(folds.repeticion == rep) &
                                  (folds.fold == fold) &
                                  (folds.split == "valid")])
            malas += len(reales & va)
        chequear(malas == 0,
                 f"{metodo}: ninguna fila de validación aparece en su train "
                 f"(contaminadas: {malas})")

        # Las columnas deben coincidir con el original, o R falla al estimar.
        faltan = set(original.columns) - set(train.columns)
        chequear(not faltan, f"{metodo}: no faltan columnas del original {sorted(faltan)}")

        # Sin NaN: Apollo los propaga silenciosamente a la verosimilitud.
        cols_datos = [c for c in original.columns if c != "row_id"]
        n_nan = int(train[cols_datos].isna().sum().sum())
        chequear(n_nan == 0, f"{metodo}: sin NaN en las columnas de datos ({n_nan})")

        # El balanceo por sobremuestreo solo tiene sentido en los métodos que
        # GENERAN filas. `original` no genera nada, y `class_weights` corrige
        # el desbalance ponderando la verosimilitud (columna `peso`), no
        # igualando los conteos: su distribución de clases debe seguir siendo
        # la original.
        n_sinteticas = int(train["is_synthetic"].sum())
        if n_sinteticas > 0:
            g0 = train[(train.repeticion == 0) & (train.fold == 0)]
            conteo = g0[choice].value_counts()
            razon = conteo.max() / conteo.min()
            chequear(razon < 1.05,
                     f"{metodo}: clases balanceadas en la partición 0 "
                     f"(razón max/min = {razon:.2f})")
        elif "peso" in train.columns and train["peso"].nunique() > 1:
            # Método de ponderación: se verifica que los pesos sean el inverso
            # de la frecuencia, es decir que clase * peso sea constante.
            g0 = train[(train.repeticion == 0) & (train.fold == 0)]
            masa = g0.groupby(choice)["peso"].sum()
            razon = masa.max() / masa.min()
            chequear(razon < 1.05,
                     f"{metodo}: masa ponderada igualada entre clases "
                     f"(razón max/min = {razon:.2f})")
        else:
            print(f"  [--] {metodo}: no genera filas ni pondera, "
                  "sin comprobación de balanceo")

    print()
    if fallos:
        print(f"{len(fallos)} fallo(s) de contrato.")
        return 1
    print("Contrato verificado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())