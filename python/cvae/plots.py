"""Gráficos del espacio latente, incluyendo las elipses de confianza (§2.1a)."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .latent import LatentEllipse


def _ellipse_path(e: LatentEllipse, n: int = 200) -> np.ndarray:
    """Contorno de la elipse de confianza en el plano latente."""
    vals, vecs = np.linalg.eigh(e.cov)
    t = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    return (e.mean[:, None] + vecs @ np.diag(np.sqrt(vals * e.threshold)) @ circ).T


def latent_scatter(mu, y, ellipses: Optional[Dict[int, LatentEllipse]] = None,
                   title: str = "", label_map: Optional[Dict[int, int]] = None,
                   figsize=(9, 8)):
    """
    Scatter del espacio latente + elipses de muestreo superpuestas.

    Esta figura reemplaza a la del paper: antes las muestras sintéticas eran un
    SEGMENTO de recta dentro de cada nube; ahora se ve la región efectivamente
    muestreada.
    """
    y = np.asarray(y)
    inv = {v: k for k, v in (label_map or {}).items()}

    fig, ax = plt.subplots(figsize=figsize)
    for cls in np.unique(y):
        m = mu[y == cls]
        etiqueta = inv.get(int(cls), int(cls))
        p = ax.scatter(m[:, 0], m[:, 1], s=8, alpha=0.5, label=f"modo {etiqueta}")
        if ellipses and int(cls) in ellipses:
            borde = _ellipse_path(ellipses[int(cls)])
            ax.plot(borde[:, 0], borde[:, 1], lw=2, color=p.get_facecolor()[0])

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def compare_distributions(real, synth, cols, target_col: Optional[str] = None,
                          bins: int = 30, figsize=(4, 2.6)):
    """
    §2.3(a) — Histogramas real vs. sintético por variable y por modo.
    Devuelve la figura; una fila por variable, una columna por modo.
    """
    modos = ([None] if target_col is None
             else sorted(set(real[target_col]) & set(synth[target_col])))
    fig, axes = plt.subplots(
        len(cols), len(modos),
        figsize=(figsize[0] * len(modos), figsize[1] * len(cols)),
        squeeze=False,
    )
    for i, col in enumerate(cols):
        for j, modo in enumerate(modos):
            ax = axes[i][j]
            r = real if modo is None else real[real[target_col] == modo]
            s = synth if modo is None else synth[synth[target_col] == modo]
            rango = (min(r[col].min(), s[col].min()), max(r[col].max(), s[col].max()))
            ax.hist(r[col], bins=bins, range=rango, density=True, alpha=0.55, label="real")
            ax.hist(s[col], bins=bins, range=rango, density=True, alpha=0.55, label="sintético")
            if i == 0:
                ax.set_title("global" if modo is None else f"modo {modo}")
            if j == 0:
                ax.set_ylabel(col)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
