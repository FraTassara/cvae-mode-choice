"""
Visualización del espacio latente por proyección PCA, para n_z > 2.

Cuando el espacio latente tiene más de 2 dimensiones no se puede dibujar
directamente. Se proyecta a las dos primeras componentes principales.

Se usa PCA (lineal) a propósito, NO UMAP/t-SNE: la proyección lineal preserva
la estructura elíptica gaussiana en la que se basa el muestreo, de modo que una
elipse en n_z dimensiones se proyecta a una elipse en 2D. Así la figura es
comparable con la del CVAE base (n_z = 2). El % de varianza explicada por las
dos componentes se anota en el título para que la pérdida de información sea
explícita.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

AZUL = "#1F3A5C"


def _pca_2d(mu: np.ndarray):
    """PCA a 2D. Devuelve (proyección, matriz_componentes, media, var_explicada)."""
    media = mu.mean(axis=0)
    centrado = mu - media
    # SVD para la PCA (estable y sin dependencias extra).
    U, S, Vt = np.linalg.svd(centrado, full_matrices=False)
    comp = Vt[:2]                       # (2, n_z)
    proy = centrado @ comp.T            # (n, 2)
    var = (S ** 2) / (len(mu) - 1)
    var_expl = var[:2].sum() / var.sum()
    return proy, comp, media, float(var_expl)


def _proyectar_elipse(elipse, comp, media, n=200):
    """
    Proyecta el contorno de una elipse de confianza n_z-dimensional a 2D.

    Como la proyección es lineal, se muestrea el borde del elipsoide en el
    espacio original y se proyecta. El borde se genera a partir de la
    descomposición de la covarianza en las dos direcciones principales de la
    PROPIA elipse, escaladas al umbral chi-cuadrado.
    """
    # Direcciones principales de la elipse (en n_z dims) y sus longitudes.
    vals, vecs = np.linalg.eigh(elipse.cov)
    k = np.sqrt(elipse.threshold)
    t = np.linspace(0, 2 * np.pi, n)
    # Círculo unitario -> elipsoide (usando las 2 direcciones de mayor varianza,
    # que son las que dominan el contorno visible).
    ejes = vecs[:, -2:] * (k * np.sqrt(vals[-2:]))     # (n_z, 2)
    borde_nz = elipse.mean + (np.cos(t)[:, None] * ejes[:, 0]
                              + np.sin(t)[:, None] * ejes[:, 1])
    # Proyectar a las componentes PCA globales.
    return (borde_nz - media) @ comp.T


def latent_scatter_pca(mu: np.ndarray, y, ellipses: Optional[Dict] = None,
                       label_map: Optional[Dict[int, int]] = None,
                       title: str = "espacio latente (PCA 2D)"):
    """
    Scatter del espacio latente proyectado a 2D con PCA, con las elipses de cada
    clase proyectadas. Firma análoga a cvae.plots.latent_scatter.
    """
    y = np.asarray(y)
    proy, comp, media, var_expl = _pca_2d(mu)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    clases = sorted(np.unique(y))
    colores = plt.cm.tab10(np.linspace(0, 1, max(len(clases), 3)))

    inv = {v: k for k, v in (label_map or {}).items()}
    for color, cls in zip(colores, clases):
        m = proy[y == cls]
        etiqueta = f"modo {inv.get(int(cls), cls)}" if inv else f"clase {cls}"
        ax.scatter(m[:, 0], m[:, 1], s=14, color=color, alpha=0.6,
                   label=etiqueta, zorder=2)
        if ellipses and int(cls) in ellipses:
            borde = _proyectar_elipse(ellipses[int(cls)], comp, media)
            ax.plot(borde[:, 0], borde[:, 1], color=color, lw=1.8, zorder=3)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{title}\n(2 PC explican {var_expl*100:.0f}% de la varianza)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
