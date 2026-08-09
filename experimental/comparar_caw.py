#!/usr/bin/env python3
"""
Compara el CVAE_caw (experimental, KL por clase) con el CVAE actual del pipeline.

Lee las estimaciones de ambos —el CVAE actual desde results/, el CVAE_caw desde
experimental_results/— y arma una tabla lado a lado con las métricas que de
verdad importan: accuracy, f1_macro, recall de la clase minoritaria y nº de
coeficientes con signo invertido.

Esto cierra la cadena: muestra si la mejora del JSD (materia prima) se traduce
en mejores métricas del MNL (producto final).

NO escribe en results/. Solo lee y muestra (y guarda un CSV en
experimental_results/).

Uso:
    python experimental/comparar_caw.py --dataset sm_centro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "python"))
from cvae.config_repo import cargar  # noqa: E402


def _resumir(df, metodo, choice_col):
    """Media entre particiones de las métricas clave, para un método."""
    sub = df[df.metodo == metodo]
    if "convergio" in sub:
        sub = sub[sub.convergio.astype(bool)]
    if sub.empty:
        return None
    out = {"metodo": metodo}
    for col, nombre in [("accuracy", "accuracy"), ("f1_macro", "f1_macro"),
                        ("loglik_media", "loglik")]:
        if col in sub:
            out[nombre] = round(sub[col].mean(), 4)
    # recall de la clase minoritaria: la columna recall_* con menor soporte medio
    rec = [c for c in sub.columns if c.startswith("recall_")]
    if rec:
        medias = {c: sub[c].mean() for c in rec}
        # la minoritaria suele ser la de menor recall en 'original'; aquí solo
        # reportamos todas para que se vea el detalle.
        for c in sorted(rec):
            out[c] = round(medias[c], 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    cfg = cargar()
    ds = args.dataset
    choice = cfg.dataset(ds)["choice_col"]

    # CVAE actual: de results/estimaciones_<ds>.csv
    est_actual = cfg.rutas.results / f"estimaciones_{ds}.csv"
    # CVAE_caw: de experimental_results/
    exp_dir = RAIZ / "experimental_results"
    est_caw = exp_dir / f"estimaciones_{ds}_CVAE_caw.csv"

    if not est_actual.exists():
        raise SystemExit(f"Falta {est_actual} (corre el pipeline normal).")
    if not est_caw.exists():
        raise SystemExit(
            f"Falta {est_caw}. Corre antes:\n"
            f"  python experimental/exp_cv_caw.py --dataset {ds}\n"
            f"  Rscript experimental/estimar_caw.R --dataset {ds}")

    df_actual = pd.read_csv(est_actual)
    df_caw = pd.read_csv(est_caw)

    filas = []
    r = _resumir(df_actual, "CVAE", choice)
    if r:
        filas.append(r)
    r = _resumir(df_caw, "CVAE_caw", choice)
    if r:
        filas.append(r)

    tabla = pd.DataFrame(filas)
    dest = exp_dir / f"comparacion_caw_{ds}.csv"
    tabla.to_csv(dest, index=False)

    print(f"\n=== {ds}: CVAE actual vs CVAE_caw (KL por clase) ===\n")
    print(tabla.to_string(index=False))
    print(f"\n-> {dest}")
    print("\nLee: si f1_macro y el recall de la clase minoritaria suben en")
    print("CVAE_caw, la mejora del JSD se tradujo en mejor desempeño del MNL.")
    print("Si la accuracy baja, es lo esperado (balancear sacrifica accuracy).")


if __name__ == "__main__":
    main()
