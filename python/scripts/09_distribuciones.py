#!/usr/bin/env python3
"""
Paso 9 — §2.3(a): figuras de distribución real vs. sintético, por modo.

El JSD (paso 6) resume en un número cuánto difiere cada variable; estas figuras
muestran DÓNDE está la diferencia — si el sintético desplaza la media, aplana
una cola, o pierde una masa en cero. Es el complemento visual que pide el plan:
"extenderlo con visualizaciones por modo y por variable".

Genera un PNG por método (ROS, SMOTE, CVAE): una fila por atributo principal,
una columna por modo, con las densidades real y sintético superpuestas.

Uso:
    python python/scripts/09_distribuciones.py --dataset sm_centro
    python python/scripts/09_distribuciones.py --dataset sm_centro --vars TVIA1,CTOT1_w
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # sin ventana; solo escribe archivos
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae import plots  # noqa: E402
from cvae.config_repo import cargar  # noqa: E402


def _vars_principales(cfg, ds: str) -> list[str]:
    """
    Atributos que entran en alguna utilidad del MNL: tiempos, costos, esperas.
    Son los que el plan pide visualizar (tiempo de viaje, costo, tiempo de
    espera), y los que importan para las reversiones de signo.
    """
    d = cfg.dataset(ds)
    cols: list[str] = []
    for alt in d["modelo"]["alternativas"]:
        for v in alt["atributos"].values():
            if v not in cols:
                cols.append(v)
    return cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--vars", default=None,
                    help="lista separada por comas; por defecto, los atributos "
                         "del modelo")
    ap.add_argument("--bins", type=int, default=30)
    args = ap.parse_args()

    cfg = cargar()
    ds = args.dataset
    d = cfg.dataset(ds)
    choice = d["choice_col"]

    original = pd.read_csv(cfg.ruta_original(ds))

    if args.vars:
        cols = [c.strip() for c in args.vars.split(",")]
    else:
        cols = _vars_principales(cfg, ds)
    faltan = [c for c in cols if c not in original.columns]
    if faltan:
        raise SystemExit(f"No están en el dataset: {faltan}")

    destino_dir = cfg.rutas.results / "figuras"
    destino_dir.mkdir(parents=True, exist_ok=True)

    generados = []
    for metodo in cfg.metodos:
        ruta = cfg.ruta_train(ds, metodo)
        if not ruta.exists():
            continue
        train = pd.read_csv(ruta, compression="gzip")
        synth = train[train.is_synthetic == 1]
        if synth.empty:
            # original y class_weights no generan datos: nada que comparar.
            continue

        fig = plots.compare_distributions(
            original, synth, cols, target_col=choice, bins=args.bins)
        fig.suptitle(f"{ds} — {metodo}: real vs. sintético por modo", y=1.002)
        salida = destino_dir / f"dist_{ds}_{metodo}.png"
        fig.savefig(salida, dpi=130, bbox_inches="tight")
        generados.append(salida)
        print(f"  {metodo}: {salida.name}")

    if not generados:
        raise SystemExit(
            "Ningún método con sintéticos. Corre primero:\n"
            f"  python python/scripts/02_sinteticos.py --dataset {ds}")

    print(f"\n{len(generados)} figura(s) en {destino_dir}/")
    print("Cada fila es un atributo; cada columna, un modo. Azul = real, "
          "naranja = sintético.")
    print("Busca dónde el sintético desplaza la media o aplana una cola: ahí")
    print("se origina la distorsión que el JSD resume y que mueve los signos.")


if __name__ == "__main__":
    main()
