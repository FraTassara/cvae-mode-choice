#!/usr/bin/env python3
"""
Paso 4 — Agregar los resultados que produjo R y armar las tablas finales.

Entrada:  results/estimaciones_{ds}.csv   (lo escribe R/03_estimar.R)
Salidas:  results/tabla_predictiva_{ds}.csv
          results/tabla_coeficientes_{ds}.csv
          results/tabla_signos_{ds}.csv

Uso:
    python python/scripts/04_reporte.py --dataset lc_centro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import cargar  # noqa: E402

METRICAS = [
    "accuracy", "f1_macro", "loglik_media", "prob_media_observada",
    "l1_market_share", "vot", "rho2",
]


def tabla_predictiva(df: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación ENTRE particiones para cada métrica."""
    cols = [c for c in METRICAS if c in df.columns]
    cols += [c for c in df.columns if c.startswith(("f1_", "recall_", "precision_"))
             and c != "f1_macro"]
    agg = df.groupby("metodo")[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{'media' if s == 'mean' else 'sd'}" for c, s in agg.columns]
    return agg.round(4)


def tabla_coeficientes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Media del coeficiente, S.D. entre particiones y S.E. promedio del estimador.

    Son cosas distintas y conviene reportar ambas: la S.D. mide inestabilidad
    ante la partición; el S.E. mide precisión de la estimación dada la partición.
    """
    betas = sorted(c for c in df.columns if c.startswith("beta_"))
    filas = []
    for metodo, g in df.groupby("metodo"):
        for b in betas:
            nombre = b[len("beta_"):]
            se_col = f"se_{nombre}"
            v = g[b].dropna()
            if v.empty:
                continue
            filas.append({
                "metodo": metodo,
                "coeficiente": nombre,
                "media": v.mean(),
                "sd_entre_particiones": v.std(),
                "se_promedio_estimador": g[se_col].mean() if se_col in g else np.nan,
                "n_particiones": len(v),
            })
    return pd.DataFrame(filas).round(6)


def tabla_signos(df: pd.DataFrame, esperados: dict) -> pd.DataFrame:
    """Cuántas particiones dan el signo contrario al que predice la teoría."""
    filas = []
    for metodo, g in df.groupby("metodo"):
        for nombre, signo in esperados.items():
            col = f"beta_{nombre}"
            if col not in g:
                continue
            v = g[col].dropna()
            if v.empty:
                continue
            filas.append({
                "metodo": metodo,
                "coeficiente": nombre,
                "signo_esperado": signo,
                "media": v.mean(),
                "pct_particiones_invertido": (np.sign(v) != signo).mean(),
                "invertido_en_media": bool(np.sign(v.mean()) != signo),
            })
    return pd.DataFrame(filas).round(4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    cfg = cargar()
    origen = cfg.ruta_estimaciones(args.dataset)
    if not origen.exists():
        raise SystemExit(f"Falta {origen}. Corre `make estimar` primero.")

    df = pd.read_csv(origen)
    if "convergio" in df:
        n_malos = int((~df.convergio.astype(bool)).sum())
        if n_malos:
            print(f"AVISO: {n_malos}/{len(df)} estimaciones no convergieron; "
                  "se excluyen de las tablas.")
            df = df[df.convergio.astype(bool)]

    esperados = cfg.dataset(args.dataset)["modelo"].get("signos_esperados", {}) or {}

    salidas = {
        "tabla_predictiva": tabla_predictiva(df),
        "tabla_coeficientes": tabla_coeficientes(df),
        "tabla_signos": tabla_signos(df, esperados),
    }

    for nombre, tabla in salidas.items():
        destino = cfg.rutas.results / f"{nombre}_{args.dataset}.csv"
        tabla.to_csv(destino, index=(nombre == "tabla_predictiva"))
        print(f"\n--- {nombre} ---")
        print(tabla.to_string())
        print(f"-> {destino.name}")


if __name__ == "__main__":
    main()
