#!/usr/bin/env python3
"""
Experimento con validación cruzada: CVAE con KL por clase (CAW) vs. el pipeline.

Genera datos sintéticos con el CVAE mejorado (kl_min por clase) usando LA MISMA
validación cruzada 5x10 del pipeline, para que sus métricas sean comparables con
las del CVAE actual. A diferencia de exp_comparar.py (que era un punto único
sobre todo el dataset), esto respeta las particiones y produce un método nuevo,
`CVAE_caw`, estimable con el R de siempre.

AISLAMIENTO: todo lo nuevo va a experimental_results/, con nombres propios. NO
escribe en data/processed/ ni en results/, así que no puede sobrescribir ningún
resultado previo.

Uso:
    python experimental/exp_cv_caw.py --dataset sm_centro
    python experimental/exp_cv_caw.py --dataset lc_centro --kl-min 0.2

Después, para estimar el MNL sobre estos datos (con tu R de siempre), apunta la
estimación al CSV que este script deja en experimental_results/. Ver README.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "python"))
sys.path.insert(0, str(RAIZ))

from cvae.config_repo import cargar  # noqa: E402

METODO_NOMBRE = "CVAE_caw"   # etiqueta del método nuevo (no colisiona con CVAE)


def _generar_caw_particion(train, cfg, ds, semilla, kl_min):
    """
    Igual que pipeline.generar_sinteticos_cvae, pero con el CVAE class-aware.
    Entrena un modelo por partición sobre SUS filas de train y genera sintéticos.
    """
    import tensorflow as tf
    from tensorflow.keras import backend as K
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    K.clear_session()

    from cvae.data import prepare_from_frame
    from cvae.generate import generate
    from cvae.pipeline import configs_desde_yaml
    from experimental.model_class_aware import (build_cvae_class_aware,
                                                pesos_kl_por_clase)

    dcfg, mcfg, gcfg = configs_desde_yaml(cfg, ds, semilla)

    columnas = [c for c in train.columns if c != "row_id"]
    ds_obj = prepare_from_frame(train[columnas], dcfg)

    y_int = np.argmax(ds_obj.y_cat, axis=1)
    pesos = pesos_kl_por_clase(y_int, kl_min)

    modelo = build_cvae_class_aware(ds_obj.n_x, ds_obj.n_y, mcfg,
                                    peso_por_clase=pesos)
    modelo.fit(ds_obj.X_scaled, ds_obj.y_cat,
               validation_data=ds_obj.validation_data, verbose=0)

    synth, _ = generate(modelo, ds_obj, gcfg, dcfg)
    synth["row_id"] = -1
    return synth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--kl-min", type=float, default=0.2,
                    help="peso KL de la clase minoritaria (óptimo práctico 0.2)")
    args = ap.parse_args()

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    cfg = cargar()
    cfg.verificar_manifiesto()
    ds = args.dataset

    # ----- carpeta aislada para resultados experimentales -----
    exp_dir = RAIZ / "experimental_results"
    exp_dir.mkdir(exist_ok=True)

    original = pd.read_csv(cfg.ruta_original(ds))
    folds = pd.read_csv(cfg.ruta_folds(ds))
    choice = cfg.dataset(ds)["choice_col"]

    print(f"[{ds}] generando {METODO_NOMBRE} (kl_min={args.kl_min}) "
          f"con validación cruzada 5x10")
    print(f"  salida aislada -> experimental_results/\n")

    salida = []
    particiones = folds[folds.split == "train"].groupby(["repeticion", "fold"])
    total = particiones.ngroups
    t0 = time.time()

    for k, ((rep, fold), g) in enumerate(particiones, start=1):
        train = original[original.row_id.isin(g.row_id)].reset_index(drop=True)
        semilla = int(cfg.semillas["oversampling"] * 10_000 + rep * 100 + fold)

        sint = _generar_caw_particion(train, cfg, ds, semilla, args.kl_min)

        train = train.assign(is_synthetic=0)
        if len(sint):
            sint = sint.assign(is_synthetic=1)
        bloque = pd.concat([train, sint], ignore_index=True)
        bloque["peso"] = 1.0
        bloque.insert(0, "fold", fold)
        bloque.insert(0, "repeticion", rep)
        salida.append(bloque)

        if k % 10 == 0 or k == total:
            print(f"    {k}/{total} particiones  ({time.time() - t0:.0f}s)")

    df = pd.concat(salida, ignore_index=True)
    destino = exp_dir / f"{ds}_{METODO_NOMBRE}_train.csv.gz"
    df.to_csv(destino, index=False, compression="gzip")

    n_sint = int(df.is_synthetic.sum())
    print(f"\n    {len(df)} filas ({n_sint} sintéticas) -> {destino.name}")
    print(f"\nGenerado en experimental_results/. Para estimar el MNL y comparar")
    print(f"con tu CVAE actual, ver experimental/README.md (sección 'CV completa').")


if __name__ == "__main__":
    main()
