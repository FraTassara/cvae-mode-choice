#!/usr/bin/env python3
"""
Barrido de kl_min: encuentra el peso KL óptimo para la clase minoritaria.

El experimento anterior mostró que el KL por clase (con n_z=2) es la mejora que
funciona: bajó el JSD de la minoritaria de 0.196 a 0.166 con kl_min=0.3. Este
script recorre varios valores de kl_min para ver cuál es el mejor y si hay un
punto donde deja de ayudar.

n_z se mantiene en 2 (el experimento mostró que subirlo perjudica con clases
pequeñas). Solo varía kl_min.

Reutiliza las funciones de exp_comparar.py; no duplica lógica.

Uso:
    python experimental/barrido_kl.py --dataset sm_centro
    python experimental/barrido_kl.py --dataset sm_centro --valores 1.0,0.5,0.3,0.2,0.1
"""

from __future__ import annotations

import argparse
import os
import sys
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="sm_centro")
    ap.add_argument("--valores", default="1.0,0.5,0.3,0.2,0.1",
                    help="valores de kl_min separados por comas")
    ap.add_argument("--n-z", type=int, default=2,
                    help="dimensión latente (fija; 2 por defecto)")
    args = ap.parse_args()

    valores = [float(v) for v in args.valores.split(",")]

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    from dataclasses import replace

    from cvae.data import prepare_from_frame
    from cvae.latent import fit_ellipses
    from cvae.pipeline import configs_desde_yaml
    from experimental.model_class_aware import (build_cvae_class_aware,
                                                pesos_kl_por_clase)
    # Reutilizamos las funciones del experimento, sin duplicarlas.
    from experimental.exp_comparar import (_jsd_minoritaria,
                                           _entrenar_y_muestrear)

    cfg = cargar()
    ds = args.dataset
    d = cfg.dataset(ds)
    choice = d["choice_col"]

    datos = pd.read_csv(cfg.ruta_original(ds))
    columnas = [c for c in datos.columns if c != "row_id"]
    dcfg, mcfg_base, gcfg = configs_desde_yaml(
        cfg, ds, cfg.semillas["cvae_muestreo"])
    dataset = prepare_from_frame(datos[columnas], dcfg)

    y_int_all = np.argmax(dataset.y_cat, axis=1)
    clase_min_int = int(pd.Series(y_int_all).value_counts().idxmin())
    inv_label = {v: k for k, v in dataset.label_map.items()}
    clase_min_real = inv_label[clase_min_int]
    ignorar = {"row_id", "is_synthetic", "repeticion", "fold", "peso"}
    cols_datos = [c for c in datos.columns
                  if c not in ignorar and c != choice]

    print(f"[{ds}] barrido de kl_min con n_z={args.n_z}")
    print(f"  clase minoritaria: interna={clase_min_int}, real={clase_min_real}")
    print(f"  valores: {valores}\n")

    mcfg = replace(mcfg_base, n_z=args.n_z)
    filas = []
    for kl_min in valores:
        pesos = pesos_kl_por_clase(y_int_all, kl_min)
        peso_min = round(pesos[clase_min_int], 3)
        synth, mu, y = _entrenar_y_muestrear(
            build_cvae_class_aware, dataset, mcfg, gcfg, dcfg,
            peso_por_clase=pesos)
        jsd = _jsd_minoritaria(datos, synth, choice, clase_min_real, cols_datos)
        filas.append({"kl_min": kl_min, "peso_clase_min": peso_min,
                      "jsd_minoritaria": round(jsd, 4)})
        print(f"  kl_min={kl_min:<4} (peso min={peso_min:<5}) "
              f"-> JSD minoritaria = {jsd:.4f}")

    tabla = pd.DataFrame(filas)
    base_jsd = tabla.loc[tabla.kl_min == 1.0, "jsd_minoritaria"]
    ref = float(base_jsd.iloc[0]) if len(base_jsd) else tabla.jsd_minoritaria.max()
    tabla["mejora_vs_base"] = (ref - tabla.jsd_minoritaria).round(4)

    dest = cfg.rutas.results / f"barrido_kl_{ds}.csv"
    tabla.to_csv(dest, index=False)

    print("\n--- Resumen del barrido ---")
    print(tabla.to_string(index=False))

    mejor = tabla.loc[tabla.jsd_minoritaria.idxmin()]
    print(f"\nMejor kl_min = {mejor.kl_min} "
          f"(JSD minoritaria {mejor.jsd_minoritaria}, "
          f"mejora {mejor.mejora_vs_base:+.4f} vs base)")
    print("Un JSD que baja y luego sube indica un óptimo intermedio; si baja")
    print("monótonamente, el mínimo probado es el mejor y podría probarse más.")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
