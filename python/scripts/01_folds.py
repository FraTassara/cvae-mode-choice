#!/usr/bin/env python3
"""
Paso 1 — Normalizar los datos crudos y construir los folds.

Salidas (el "contrato" que consume R):

  data/processed/{ds}_original.csv
      El dataset original ya filtrado (clases válidas, filtros de config.yaml),
      con una columna `row_id` 0..n-1. R usa ESTE archivo, no el crudo, para
      garantizar que ambos lados filtran idéntico.

  data/folds/{ds}_folds.csv
      Columnas: row_id, repeticion, fold, split ("train" | "valid").
      Los folds se construyen UNA sola vez, con la semilla de config.yaml, y
      los usan tanto Python (para generar sintéticos) como R (para estimar).

Uso:
    python python/scripts/01_folds.py                # todos los datasets
    python python/scripts/01_folds.py --dataset lc_centro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import cargar  # noqa: E402


def normalizar(cfg, ds: str) -> pd.DataFrame:
    """Lee el crudo, aplica filtros y añade row_id."""
    d = cfg.dataset(ds)
    df = pd.read_csv(cfg.ruta_raw(ds), sep=d["sep"])

    for expr in d.get("filtros", []):
        antes = len(df)
        df = df.query(expr)
        print(f"    filtro '{expr}': {antes} -> {len(df)} filas")

    choice = d["choice_col"]
    df = df[df[choice].isin(d["clases"])]

    df = df.reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))
    return df


def construir_folds(df: pd.DataFrame, cfg, ds: str) -> pd.DataFrame:
    """
    Folds sobre los datos ORIGINALES. El sobremuestreo se aplica después,
    solo dentro de cada partición de entrenamiento.
    """
    v = cfg.validacion
    y = df[cfg.dataset(ds)["choice_col"]]

    Splitter = RepeatedStratifiedKFold if v["estratificado"] else RepeatedKFold
    splitter = Splitter(n_splits=v["n_splits"], n_repeats=v["n_repeats"],
                        random_state=cfg.semillas["folds"])

    filas = []
    for i, (idx_tr, idx_va) in enumerate(splitter.split(df, y)):
        rep, fold = divmod(i, v["n_splits"])
        for idx, split in ((idx_tr, "train"), (idx_va, "valid")):
            filas.append(pd.DataFrame({
                "row_id": df["row_id"].to_numpy()[idx],
                "repeticion": rep,
                "fold": fold,
                "split": split,
            }))
    return pd.concat(filas, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=None, help="uno solo; por defecto todos")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = cargar(args.config) if args.config else cargar()
    cfg.rutas.crear()

    objetivos = [args.dataset] if args.dataset else cfg.datasets
    resumen = {}

    for ds in objetivos:
        print(f"\n[{ds}]")
        crudo = cfg.ruta_raw(ds)
        if not crudo.exists():
            print(f"    SALTADO: falta {crudo.relative_to(cfg.rutas.raw.parent.parent)}")
            continue

        df = normalizar(cfg, ds)
        df.to_csv(cfg.ruta_original(ds), index=False)
        print(f"    {len(df)} filas -> {cfg.ruta_original(ds).name}")
        print(f"    clases: {df[cfg.dataset(ds)['choice_col']].value_counts().sort_index().to_dict()}")

        folds = construir_folds(df, cfg, ds)
        folds.to_csv(cfg.ruta_folds(ds), index=False)
        n_part = folds.groupby(["repeticion", "fold"]).ngroups
        print(f"    {n_part} particiones -> {cfg.ruta_folds(ds).name}")

        resumen[ds] = {"n_filas": len(df), "n_particiones": int(n_part)}

    cfg.escribir_manifiesto({"paso_01": resumen})
    print(f"\nManifiesto actualizado (config_hash = {cfg.hash()})")


if __name__ == "__main__":
    main()
