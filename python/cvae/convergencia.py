"""
Política única sobre `convergio` para todas las tablas que agregan particiones.

`convergio` es el `successfulEstimation` de Apollo, que marca "false
convergence" aunque se alcance el mismo log-likelihood (ver
docs/diagnostico_corrigendum.md §2). Excluir esas filas hacía que cada método
se promediara sobre un subconjunto distinto de particiones (en SM-Centro,
class_weights sobre 11 de 50).

Regla: entran TODAS las particiones estimadas, igual para todos los métodos.
Solo se descarta una estimación sin coeficientes finitos, y se informa.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def aplicar_politica(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve (filas que entran, conteo por método).

    El conteo tiene, por método: `n_particiones` (las que entran),
    `n_no_convergidas` (entre ellas, las que Apollo marcó como no convergidas)
    y `n_descartadas` (sin coeficientes finitos; no entran).
    """
    betas = [c for c in df.columns if c.startswith("beta_")]
    validas = pd.Series(True, index=df.index)
    for _, g in df.groupby("metodo"):
        # Solo los coeficientes de la especificación de ese método: el CSV
        # acumulado puede traer columnas de otros métodos llenas de NA.
        b = [c for c in betas if g[c].notna().any()]
        if b:
            validas.loc[g.index] = np.isfinite(g[b].to_numpy(dtype=float)).all(axis=1)

    conv = (df["convergio"].astype(bool) if "convergio" in df
            else pd.Series(True, index=df.index))
    conteo = pd.DataFrame({
        "n_particiones": validas.groupby(df["metodo"]).sum(),
        "n_no_convergidas": (~conv & validas).groupby(df["metodo"]).sum(),
        "n_descartadas": (~validas).groupby(df["metodo"]).sum(),
    }).astype(int)
    conteo.index.name = "metodo"

    if conteo["n_descartadas"].any():
        print("AVISO: estimaciones sin coeficientes finitos, descartadas:\n"
              + conteo.loc[conteo.n_descartadas > 0, ["n_descartadas"]].to_string())
    if conteo["n_particiones"].nunique() > 1:
        print("AVISO: los métodos no se promedian sobre el mismo número de "
              "particiones:\n" + conteo[["n_particiones"]].to_string())
    return df[validas], conteo
