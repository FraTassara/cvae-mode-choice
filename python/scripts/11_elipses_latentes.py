#!/usr/bin/env python3
"""
Paso 11 — Figuras del espacio latente con elipses de confianza (§2.1a / §3.4).

Reproduce la Figura 3 del paper (scatter del espacio latente 2D con la elipse
de confianza de cada modo superpuesta), pero con el código corregido: las
elipses son las que efectivamente se usan para el muestreo estocástico, no una
ilustración.

Entrena un CVAE sobre TODOS los datos reales del dataset (sin partición: esta
figura es descriptiva del espacio latente, no una evaluación), codifica cada
observación a su media latente, ajusta una elipse por clase y las dibuja.

Requiere n_z = 2 (si la arquitectura usa más dimensiones, la figura no es
representable y el script lo avisa).

Uso:
    python python/scripts/11_elipses_latentes.py --dataset sm_centro
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import cargar  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--confianza", type=float, default=None,
                    help="un solo nivel; por defecto genera 0.70/0.80/0.90/0.95")
    args = ap.parse_args()

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    from cvae import plots
    from cvae.data import prepare_from_frame
    from cvae.latent import fit_ellipses
    from cvae.model import build_cvae
    from cvae.pipeline import configs_desde_yaml
    from cvae.seeding import set_all_seeds

    cfg = cargar()
    ds = args.dataset
    d = cfg.dataset(ds)

    origen = cfg.ruta_original(ds)
    if not origen.exists():
        raise SystemExit(
            f"Falta {origen}. Corre primero:\n"
            f"  python python/scripts/01_folds.py --dataset {ds}")

    dcfg, mcfg, gcfg = configs_desde_yaml(cfg, ds, cfg.semillas["cvae_muestreo"])
    if mcfg.n_z != 2:
        raise SystemExit(
            f"La figura del espacio latente requiere n_z = 2, pero la "
            f"arquitectura de '{ds}' usa n_z = {mcfg.n_z}. No es representable "
            "en el plano.")

    # Niveles a graficar: uno solo si se pide con --confianza; si no, los cuatro
    # del análisis de sensibilidad (§2.7) para ver cómo crece la región de
    # muestreo al subir el nivel.
    if args.confianza is not None:
        niveles = [args.confianza]
    else:
        niveles = [0.70, 0.80, 0.90, 0.95]

    datos = pd.read_csv(origen)
    columnas = [c for c in datos.columns if c != "row_id"]
    dataset = prepare_from_frame(datos[columnas], dcfg)

    # El CVAE se entrena UNA sola vez: el espacio latente no depende del nivel
    # de la elipse, solo cambia el tamaño del contorno. Así las cuatro figuras
    # son comparables (mismas nubes, distinto contorno).
    print(f"[{ds}] entrenando CVAE sobre {len(datos)} observaciones reales...")
    set_all_seeds(cfg.semillas["cvae_entrena"])
    modelo = build_cvae(dataset.n_x, dataset.n_y, mcfg)
    modelo.fit(dataset.X_scaled, dataset.y_cat, verbose=0)
    mu, _ = modelo.encode(dataset.X_scaled, dataset.y_cat)

    # `dataset.X_scaled` puede contener solo el train (prepare_from_frame parte
    # 80/20 si hay validación), mientras que `dataset.y` trae todas las filas.
    # Para que las etiquetas cuadren con `mu`, se derivan del y_cat que SÍ
    # corresponde a las filas codificadas.
    import numpy as np
    y_plot = pd.Series(np.argmax(dataset.y_cat, axis=1))
    assert len(y_plot) == len(mu), (
        f"desalineado: mu={len(mu)} vs y={len(y_plot)}")

    label_map = getattr(dataset, "label_map", None)
    destino_dir = cfg.rutas.results / "figuras"
    destino_dir.mkdir(parents=True, exist_ok=True)

    generados = []
    for confianza in niveles:
        ellipses = fit_ellipses(mu, y_plot, confidence=confianza)
        fig = plots.latent_scatter(
            mu, y_plot, ellipses=ellipses, label_map=label_map,
            title=f"{ds} — espacio latente y elipses al "
                  f"{int(confianza*100)}%")
        # Sufijo solo cuando hay varios niveles, para no romper el nombre
        # 'elipses_<ds>.png' que ya usa el informe para el 80%.
        if confianza == 0.80 and len(niveles) > 1:
            salida = destino_dir / f"elipses_{ds}.png"
        elif len(niveles) == 1:
            salida = destino_dir / f"elipses_{ds}.png"
        else:
            salida = destino_dir / f"elipses_{ds}_conf{int(confianza*100)}.png"
        fig.savefig(salida, dpi=140, bbox_inches="tight")
        generados.append(salida.name)
        print(f"  {int(confianza*100)}% -> {salida.name}")

    print(f"\n{len(generados)} figura(s) en {destino_dir}/")
    print("Cada nube es un modo; el contorno es su elipse de confianza, la "
          "región\ndonde se muestrean los sintéticos. Al subir el nivel, la "
          "elipse crece\ny alcanza zonas de menor densidad del espacio latente.")


if __name__ == "__main__":
    main()
