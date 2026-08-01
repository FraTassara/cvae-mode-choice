#!/usr/bin/env python3
"""
Paso 10 — §2.7: sensibilidad de los resultados al nivel de la elipse.

El nivel 0.80 de las elipses de confianza es una elección de diseño. R2 objetó
que es ad hoc. Este análisis regenera el CVAE a 0.70, 0.90 y 0.95, y muestra que
el desempeño predictivo NO cambia sustancialmente — lo que responde la crítica.

El paso NO reejecuta nada: LEE los resultados que ya dejaron las corridas de cada
nivel y arma la tabla comparativa. El flujo para producir esos resultados está
en la ayuda de abajo y en el README.

Uso:
    # 1. generar y estimar cada nivel (además del 0.80 que ya es 'CVAE'):
    #    for c in 0.70 0.90 0.95:
    #      python python/scripts/02_sinteticos.py --dataset sm_centro --metodo CVAE --confianza $c
    #      Rscript R/03_estimar.R --dataset sm_centro --metodo CVAE_conf70   # etc.
    # 2. consolidar:
    python python/scripts/10_sensibilidad_elipses.py --dataset sm_centro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import cargar  # noqa: E402

# El 0.80 corresponde al método 'CVAE' a secas (el nivel por defecto del config).
NIVELES = {"CVAE": 0.80, "CVAE_conf70": 0.70,
           "CVAE_conf90": 0.90, "CVAE_conf95": 0.95}

METRICAS = ["accuracy", "f1_macro", "loglik_media", "prob_media_observada"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    cfg = cargar()
    origen = cfg.ruta_estimaciones(args.dataset)
    if not origen.exists():
        raise SystemExit(f"Falta {origen}. Estima primero los niveles.")

    df = pd.read_csv(origen)
    if "convergio" in df:
        df = df[df.convergio.astype(bool)]

    presentes = [m for m in NIVELES if m in set(df.metodo)]
    faltan = [m for m in NIVELES if m not in presentes]
    if len(presentes) < 2:
        raise SystemExit(
            "Hacen falta al menos dos niveles estimados. Genera y estima:\n"
            "  --confianza 0.70 / 0.90 / 0.95 (el 0.80 es el 'CVAE' base).\n"
            f"Presentes: {presentes or 'ninguno'}")
    if faltan:
        print(f"AVISO: faltan niveles {faltan}; se comparan los presentes.")

    cols = [c for c in METRICAS if c in df.columns]
    sub = df[df.metodo.isin(presentes)].copy()
    sub["confianza"] = sub.metodo.map(NIVELES)

    tabla = (sub.groupby("confianza")[cols]
             .agg(["mean", "std"])
             .sort_index())
    tabla.columns = [f"{c}_{'media' if s == 'mean' else 'sd'}"
                     for c, s in tabla.columns]
    tabla = tabla.round(4)

    destino = cfg.rutas.results / f"sensibilidad_elipses_{args.dataset}.csv"
    tabla.to_csv(destino)

    print(f"\n--- §2.7 Sensibilidad al nivel de elipse — {args.dataset} ---")
    print(tabla.to_string())
    print(f"\n-> {destino.name}")

    # Rango de variación del f1_macro entre niveles: la cifra que resume si el
    # resultado es robusto. Si es pequeña, el 0.80 no era una elección crítica.
    if "f1_macro_media" in tabla.columns:
        rango = tabla["f1_macro_media"].max() - tabla["f1_macro_media"].min()
        base = tabla["f1_macro_media"].get(0.80, tabla["f1_macro_media"].mean())
        print(f"\nf1_macro varía {rango:.4f} entre niveles "
              f"({100 * rango / base:.1f}% del valor al 80%).")
        print("Cuanto menor, más robusto el resultado al umbral elegido.")


if __name__ == "__main__":
    main()
