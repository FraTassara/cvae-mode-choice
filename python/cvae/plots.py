"""Gráficos del espacio latente, incluyendo las elipses de confianza (§2.1a)."""

import colorsys
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .latent import LatentEllipse


# Paleta hue de ggplot2 (scales::hue_pal) para n categorías, de modo que los
# colores por modo coincidan con los de las figuras originales en R.
_GGPLOT = {
    2: ["#F8766D", "#00BFC4"],
    3: ["#F8766D", "#00BA38", "#619CFF"],
    4: ["#F8766D", "#7CAE00", "#00BFC4", "#C77CFF"],
    5: ["#F8766D", "#A3A500", "#00BF7D", "#00B0F6", "#E76BF3"],
}


def _ggplot_colors(n: int):
    if n in _GGPLOT:
        return _GGPLOT[n]
    return [colorsys.hls_to_rgb(((15 + i * (360 / n)) % 360) / 360, 0.62, 0.60)
            for i in range(n)]


def _ellipse_path(e: LatentEllipse, n: int = 200) -> np.ndarray:
    """Contorno de la elipse de confianza en el plano latente."""
    vals, vecs = np.linalg.eigh(e.cov)
    t = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    return (e.mean[:, None] + vecs @ np.diag(np.sqrt(vals * e.threshold)) @ circ).T


def latent_scatter(mu, y, ellipses: Optional[Dict[int, LatentEllipse]] = None,
                   title: str = "", label_map: Optional[Dict[int, int]] = None,
                   name_map: Optional[Dict[int, str]] = None,
                   figsize=(3.0, 3.3)):
    """
    Scatter del espacio latente + elipses de muestreo, faceteado por modo.

    Reproduce el estilo de la Figura 3 del paper (R): un panel por modo con su
    franja gris de título, fondo gris con grilla blanca, nube de color y elipse
    rellena con borde, ejes Z1/Z2 compartidos. Reemplaza a la figura del paper:
    antes las muestras sintéticas eran un SEGMENTO de recta dentro de cada nube;
    ahora se ve la región efectivamente muestreada.

    Parámetros nuevos:
      title    : título general, p.ej.
                 "SM-Centro — latent space and 80% confidence ellipses".
      name_map : {id_de_modo: nombre visible}, p.ej. {1: "Car Driver", ...}.
                 Si falta, se rotula "modo N" como antes.

    Nota: `figsize` es ahora el tamaño POR PANEL; el ancho total escala con el
    número de modos.
    """
    y = np.asarray(y)
    clases = sorted(int(c) for c in np.unique(y))
    K = len(clases)
    colors = _ggplot_colors(K)

    # label_map llega como {id_de_modo: clase_codificada}; se invierte para
    # mapear clase_codificada -> id_de_modo (igual que la versión original).
    inv = {v: k for k, v in (label_map or {}).items()}

    fig, axes = plt.subplots(1, K, figsize=(figsize[0] * K, figsize[1]),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for i, cls in enumerate(clases):
        ax = axes[i]
        modo_id = inv.get(cls, cls)
        nombre = (name_map.get(modo_id, f"modo {modo_id}")
                  if name_map else f"modo {modo_id}")

        m = mu[y == cls]
        ax.scatter(m[:, 0], m[:, 1], s=8, alpha=0.6, color=colors[i],
                   edgecolors="none")

        if ellipses and int(cls) in ellipses:
            borde = _ellipse_path(ellipses[int(cls)])
            ax.fill(borde[:, 0], borde[:, 1], color=colors[i], alpha=0.18)
            ax.plot(borde[:, 0], borde[:, 1], color=colors[i], lw=1.3)

        # Franja gris de título, como los strips de facet_wrap de ggplot.
        ax.set_title(nombre, fontsize=10, fontweight="bold", color="#3a3a3a",
                     bbox=dict(boxstyle="square,pad=0.5",
                               facecolor="#D9D9D9", edgecolor="none"))
        ax.set_facecolor("#EBEBEB")
        ax.grid(True, color="white", lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, labelsize=8)

    fig.supxlabel(r"Z$_1$", fontsize=11)
    fig.supylabel(r"Z$_2$", fontsize=11)
    if title:
        fig.suptitle(title, fontsize=12.5, fontweight="bold")
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