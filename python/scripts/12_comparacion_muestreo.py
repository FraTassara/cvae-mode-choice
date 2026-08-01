#!/usr/bin/env python3
"""
Paso 12 — Comparación del muestreo dentro de la elipse, con datos REALES.

Genera UNA figura por dataset con una fila por modo y dos columnas:
  - Izquierda: método ANTERIOR (R), muestreo uniforme sobre un cuadrado +
    filtrado por la elipse. Densidad pareja.
  - Derecha:  método ACTUAL (§2.1a), muestreo desde la gaussiana de la clase
    con rechazo por Mahalanobis. Densidad como la de los datos reales.

En cada panel se dibujan los puntos reales de la clase (gris), los sintéticos
(naranja) y el contorno de la elipse. La comparación es sobre el espacio latente
real del dataset.

El método anterior NO forma parte del pipeline: se reproduce aquí solo para la
figura comparativa del informe.

Uso:
    python python/scripts/12_comparacion_muestreo.py --dataset sm_centro
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import cargar  # noqa: E402

AZUL = "#1F3A5C"
NARANJA = "#E08A3C"
GRIS = "#C8C8C8"


def _muestreo_uniforme(elipse, n_objetivo, rng, margen=0.15):
    """
    Reproduce el método anterior (R): puntos uniformes sobre un cuadrado que
    cubre la elipse, filtrados por la distancia de Mahalanobis. Densidad pareja.
    El cuadrado se ajusta AUTOMÁTICAMENTE a la elipse (en el R original era fijo
    en [-2.5, 2.5], lo que fallaba si la clase caía fuera).
    """
    sd = np.sqrt(np.diag(elipse.cov))
    k = np.sqrt(elipse.threshold)
    lo = elipse.mean - k * sd - margen
    hi = elipse.mean + k * sd + margen

    aceptados = []
    intentos = 0
    while sum(len(a) for a in aceptados) < n_objetivo and intentos < 200:
        prop = rng.uniform(lo, hi, size=(n_objetivo * 3, elipse.n_z))
        dentro = prop[elipse.mahalanobis_sq(prop) <= elipse.threshold]
        aceptados.append(dentro)
        intentos += 1
    pts = np.vstack(aceptados) if aceptados else np.empty((0, elipse.n_z))
    return pts[:n_objetivo]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=800,
                    help="nº de puntos sintéticos por panel")
    args = ap.parse_args()

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    from cvae.data import prepare_from_frame
    from cvae.latent import fit_ellipses
    from cvae.model import build_cvae
    from cvae.pipeline import configs_desde_yaml
    from cvae.plots import _ellipse_path
    from cvae.seeding import set_all_seeds

    cfg = cargar()
    ds = args.dataset
    origen = cfg.ruta_original(ds)
    if not origen.exists():
        raise SystemExit(f"Falta {origen}. Corre antes 01_folds.py.")

    dcfg, mcfg, gcfg = configs_desde_yaml(cfg, ds, cfg.semillas["cvae_muestreo"])
    if mcfg.n_z != 2:
        raise SystemExit(f"Requiere n_z = 2; '{ds}' usa {mcfg.n_z}.")

    datos = pd.read_csv(origen)
    columnas = [c for c in datos.columns if c != "row_id"]
    dataset = prepare_from_frame(datos[columnas], dcfg)

    print(f"[{ds}] entrenando CVAE...")
    set_all_seeds(cfg.semillas["cvae_entrena"])
    modelo = build_cvae(dataset.n_x, dataset.n_y, mcfg)
    modelo.fit(dataset.X_scaled, dataset.y_cat, verbose=0)
    mu, _ = modelo.encode(dataset.X_scaled, dataset.y_cat)
    y_plot = pd.Series(np.argmax(dataset.y_cat, axis=1))

    ellipses = fit_ellipses(mu, y_plot, confidence=gcfg.confidence)
    rng = np.random.default_rng(cfg.semillas["cvae_muestreo"])

    inv = {v: k for k, v in dataset.label_map.items()}  # interno -> modo real
    # Nombre legible de cada modo, desde config.yaml.
    nombres = {int(a["id"]): a["nombre"]
               for a in cfg.dataset(ds)["modelo"]["alternativas"]}
    clases = sorted(ellipses.keys())
    n_modos = len(clases)

    # Una fila por modo, dos columnas (anterior | actual).
    fig, axes = plt.subplots(n_modos, 2,
                             figsize=(10, 4.6 * n_modos),
                             squeeze=False)

    for i, cls_interna in enumerate(clases):
        modo_real = inv[cls_interna]
        nombre = nombres.get(modo_real, f"modo {modo_real}")
        elipse = ellipses[cls_interna]
        reales = mu[y_plot == cls_interna]
        borde = _ellipse_path(elipse)

        sint_unif = _muestreo_uniforme(elipse, args.n, rng)
        sint_gauss = elipse.sample(args.n, rng)

        print(f"  {nombre} (modo {modo_real}): {len(reales)} reales, "
              f"{len(sint_unif)} unif, {len(sint_gauss)} gauss")

        # Límites del panel: centrados en la ELIPSE, no en todos los puntos
        # reales. Así el modo con elipse pequeña (p.ej. car en Swissmetro) se ve
        # a su propia escala en vez de quedar aplastado por modos más dispersos.
        # Se usa el contorno más un margen del 25%.
        cx0, cx1 = borde[:, 0].min(), borde[:, 0].max()
        cy0, cy1 = borde[:, 1].min(), borde[:, 1].max()
        mx = (cx1 - cx0) * 0.25 + 1e-6
        my = (cy1 - cy0) * 0.25 + 1e-6
        xlim = (cx0 - mx, cx1 + mx)
        ylim = (cy0 - my, cy1 + my)

        for j, (pts, titulo) in enumerate([
            (sint_unif, "anterior (R): uniforme"),
            (sint_gauss, "actual (§2.1a): gaussiano"),
        ]):
            ax = axes[i][j]
            ax.scatter(reales[:, 0], reales[:, 1], s=16, c=GRIS, alpha=0.75,
                       label="reales", zorder=1)
            ax.scatter(pts[:, 0], pts[:, 1], s=6, c=NARANJA, alpha=0.5,
                       label="sintéticos", zorder=2)
            ax.plot(borde[:, 0], borde[:, 1], color=AZUL, lw=1.8, zorder=3)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_ylabel(f"{nombre}\nz2" if j == 0 else "")
            if i == 0:
                ax.set_title(titulo, fontsize=11)
            if i == n_modos - 1:
                ax.set_xlabel("z1")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8.5, loc="best")

    fig.suptitle(f"{ds} — muestreo dentro de la elipse, por modo\n"
                 "izquierda: método anterior (no se usa) · "
                 "derecha: método actual",
                 y=1.005, fontsize=12)
    fig.tight_layout()

    destino_dir = cfg.rutas.results / "figuras"
    destino_dir.mkdir(parents=True, exist_ok=True)
    salida = destino_dir / f"comparacion_muestreo_{ds}.png"
    fig.savefig(salida, dpi=140, bbox_inches="tight")
    print(f"\n-> {salida}")
    print(f"{n_modos} modos comparados. Gris: reales; naranja: sintéticos.")
    print("La columna izquierda rellena la elipse de forma pareja; la derecha")
    print("concentra los sintéticos donde están los datos reales.")


if __name__ == "__main__":
    main()
