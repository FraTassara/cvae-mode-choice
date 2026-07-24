"""
§2.5 y §2.6 — Diagnósticos de calidad de los datos sintéticos.

  jsd_by_variable()      JSD real vs. sintético por variable (y por modo).
  jsd_internal_benchmark()  JSD entre dos mitades aleatorias de los datos reales.
                         Es la escala de referencia: un JSD sintético cercano a
                         este valor es indistinguible de la variabilidad muestral.
  oversampling_report()  nº de muestras por clase y ratio sintético/original.
  attribute_gaps()       §2.3(b): diferencias de atributos elegido - no elegido.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def _hist(a: np.ndarray, b: np.ndarray, bins: int = 30):
    """Histogramas normalizados sobre un soporte común."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return None, None
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if lo == hi:
        return np.array([1.0]), np.array([1.0])
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    eps = 1e-12
    return pa / (pa.sum() + eps) + eps, pb / (pb.sum() + eps) + eps


def jsd(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    """Jensen-Shannon divergence (base 2, en [0, 1])."""
    pa, pb = _hist(a, b, bins)
    if pa is None:
        return np.nan
    return float(jensenshannon(pa, pb, base=2) ** 2)


def jsd_by_variable(real: pd.DataFrame, synth: pd.DataFrame,
                    target_col: Optional[str] = None,
                    bins: int = 30) -> pd.DataFrame:
    """
    JSD por variable, global y desagregado por modo (§2.3a pide justamente
    la desagregación por modo, no solo el promedio).
    """
    cols = [c for c in real.columns if c in synth.columns and c != target_col]
    filas = []
    for col in cols:
        fila = {"variable": col, "jsd_global": jsd(real[col], synth[col], bins)}
        if target_col is not None:
            for m in sorted(set(real[target_col]) & set(synth[target_col])):
                fila[f"jsd_modo_{m}"] = jsd(
                    real.loc[real[target_col] == m, col],
                    synth.loc[synth[target_col] == m, col],
                    bins,
                )
        filas.append(fila)
    return pd.DataFrame(filas).set_index("variable")


def jsd_internal_benchmark(real: pd.DataFrame, target_col: Optional[str] = None,
                           n_repeats: int = 50, bins: int = 30,
                           seed: int = 0) -> pd.DataFrame:
    """
    §2.5 — Benchmark interno: JSD entre dos mitades aleatorias de los datos
    reales, repetido n_repeats veces. Da media y percentil 95 por variable.

    Interpretación para el paper: un JSD sintético por debajo del p95 de este
    benchmark es estadísticamente indistinguible del ruido de muestreo.
    """
    rng = np.random.default_rng(seed)
    cols = [c for c in real.columns if c != target_col]
    acum = {c: [] for c in cols}

    n = len(real)
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        a = real.iloc[perm[: n // 2]]
        b = real.iloc[perm[n // 2 :]]
        for c in cols:
            acum[c].append(jsd(a[c], b[c], bins))

    return pd.DataFrame(
        {
            "jsd_benchmark_media": {c: np.nanmean(v) for c, v in acum.items()},
            "jsd_benchmark_p95": {c: np.nanpercentile(v, 95) for c, v in acum.items()},
        }
    )


def oversampling_report(real: pd.DataFrame, synth: pd.DataFrame,
                        target_col: str, method: str = "CVAE") -> pd.DataFrame:
    """§2.6 — Tabla de nº de muestras sintéticas por clase y ratio."""
    r = real[target_col].value_counts().sort_index()
    s = synth[target_col].value_counts().reindex(r.index).fillna(0).astype(int)
    tab = pd.DataFrame(
        {
            "metodo": method,
            "n_original": r,
            "n_sintetico": s,
            "n_final": r + s,
            "ratio_sint_orig": (s / r).round(3),
            "share_original": (r / r.sum()).round(4),
            "share_final": ((r + s) / (r + s).sum()).round(4),
        }
    )
    tab.index.name = target_col
    return tab


def attribute_gaps(df: pd.DataFrame, target_col: str,
                   alternative_cols: Dict[int, Sequence[str]]) -> pd.DataFrame:
    """
    §2.3(b) — Diferencias medias de atributos entre la alternativa elegida y las
    no elegidas. Es lo que realmente identifica los coeficientes del MNL: si el
    sintético preserva las marginales pero destruye estos gaps, las reversiones
    de signo quedan explicadas.

    alternative_cols: {modo -> [col_tiempo, col_costo, ...]} en el MISMO orden
    para todos los modos, de forma que las posiciones sean comparables.
    """
    modos = sorted(alternative_cols)
    n_attr = len(alternative_cols[modos[0]])
    filas = []

    for modo in modos:
        sub = df[df[target_col] == modo]
        if sub.empty:
            continue
        for j in range(n_attr):
            elegido = sub[alternative_cols[modo][j]].to_numpy(dtype=float)
            otros = [
                sub[alternative_cols[m][j]].to_numpy(dtype=float)
                for m in modos
                if m != modo
            ]
            no_elegido = np.mean(otros, axis=0)
            filas.append(
                {
                    "modo_elegido": modo,
                    "atributo": j,
                    "col": alternative_cols[modo][j],
                    "media_elegido": elegido.mean(),
                    "media_no_elegido": no_elegido.mean(),
                    "gap": (elegido - no_elegido).mean(),
                }
            )
    return pd.DataFrame(filas)
