#!/usr/bin/env python3
"""
13_extraer_pendientes.py — Empaqueta los entregables del instructivo de revisores.

NO recalcula nada: lee lo que ya produjeron 04_reporte.py, 06_diagnosticos.py y
08_efectos_marginales.R, y lo reescribe con los nombres EXACTOS que pide el PDF
de pendientes, para que redacción los pegue sin fricción y sin mantener una
segunda implementación del modelo.

Entradas (en results/):
    tabla_predictiva_{ds}.csv     -> VTT (columna vot)          [04_reporte.py]
    tabla_coeficientes_{ds}.csv   -> S.E.avg                    [04_reporte.py]
    efectos_marginales_{ds}.csv          -> ef. marg. por partición [08_efectos...R]
    efectos_marginales_resumen_{ds}.csv  -> elasticidades (media/SD) [08_efectos...R]
    jsd_{ds}.csv                  -> benchmark de JSD           [06_diagnosticos]

Salidas (en results/pendientes/), con los nombres del PDF:
    tabla_vtt.csv
    tabla_se_avg_{SM,LC,SW}.csv
    tabla_elasticidades.csv
    efectos_marginales_{SM,LC,SW}_{metodo}.csv
    jsd_benchmark.csv

Uso:
    python python/scripts/13_extraer_pendientes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import cargar  # noqa: E402

# Alcance del PDF y etiquetas cortas que usa el instructivo.
DATASETS = {"sm_centro": "SM", "lc_centro": "LC", "swissmetro": "SW"}


def coef_tiempo(dmodelo: dict) -> str:
    """Nombre del coeficiente de tiempo: b_time (Swissmetro) o b_ivt (SM/LC)."""
    claves = set()
    for a in dmodelo["alternativas"]:
        claves |= set(a["atributos"].keys())
    return "b_time" if "b_time" in claves else "b_ivt"


def col_tiempo_por_alternativa(dmodelo: dict) -> dict[str, str]:
    """{id_alternativa (str): columna de tiempo de esa alternativa}."""
    tc = coef_tiempo(dmodelo)
    out = {}
    for a in dmodelo["alternativas"]:
        col = a["atributos"].get(tc)
        if col is not None:
            out[str(a["id"])] = col
    return out


def main() -> None:
    cfg = cargar()
    res = cfg.rutas.results
    out = res / "pendientes"
    out.mkdir(parents=True, exist_ok=True)

    filas_vtt, filas_elas, filas_jsd = [], [], []

    for ds, etq in DATASETS.items():
        dmodelo = cfg.dataset(ds)["modelo"]

        # ---- 1. VTT  (Revisor 2, C4) -> de tabla_predictiva_{ds}.csv --------
        f = res / f"tabla_predictiva_{ds}.csv"
        if f.exists():
            tp = pd.read_csv(f, index_col=0)      # índice = metodo
            if "vot_media" in tp.columns:
                for metodo, r in tp.iterrows():
                    filas_vtt.append({
                        "dataset": etq, "metodo": metodo,
                        "vtt": r.get("vot_media"),
                        "vtt_sd": r.get("vot_sd"),
                    })
            else:
                print(f"[{ds}] tabla_predictiva sin columna vot_media")
        else:
            print(f"[{ds}] falta {f.name}")

        # ---- 2. S.E.avg  (Revisor 1, C9) -> de tabla_coeficientes_{ds}.csv --
        f = res / f"tabla_coeficientes_{ds}.csv"
        if f.exists():
            tc = pd.read_csv(f)
            if "se_promedio_estimador" in tc.columns:
                se = (tc.pivot(index="coeficiente", columns="metodo",
                               values="se_promedio_estimador")
                        .round(6))
                se.to_csv(out / f"tabla_se_avg_{etq}.csv")
            else:
                print(f"[{ds}] tabla_coeficientes sin se_promedio_estimador")
        else:
            print(f"[{ds}] falta {f.name}")

        # ---- 3-4. Elasticidades + efectos marginales -> efectos_marginales_{ds}
        f = res / f"efectos_marginales_{ds}.csv"
        if f.exists():
            em = pd.read_csv(f)
            em["alternativa"] = em["alternativa"].astype(str)

            # 4. efectos marginales muestra a muestra: un archivo por método
            cols = ["repeticion", "fold", "convergio",
                    "covariable", "alternativa", "prob_base",
                    "prob_perturbada", "cambio_abs", "elasticidad"]
            cols = [c for c in cols if c in em.columns]
            for metodo, g in em.groupby("metodo"):
                g[cols].to_csv(
                    out / f"efectos_marginales_{etq}_{metodo}.csv", index=False)
        else:
            print(f"[{ds}] falta {f.name}")

        # 3. elasticidad PROPIA del tiempo, media y SD sobre las particiones
        #    (N9): fila cuya covariable es la columna de tiempo de su propia
        #    alternativa.
        f = res / f"efectos_marginales_resumen_{ds}.csv"
        if f.exists():
            er = pd.read_csv(f)
            er["alternativa"] = er["alternativa"].astype(str)
            tcol = col_tiempo_por_alternativa(dmodelo)
            mask = er.apply(
                lambda r: r["alternativa"] in tcol
                and r["covariable"] == tcol[r["alternativa"]], axis=1)
            for _, r in er[mask].iterrows():
                filas_elas.append({
                    "dataset": etq, "metodo": r["metodo"],
                    "alternativa": r["alternativa"],
                    "variable_tiempo": r["covariable"],
                    "elasticidad_propia_tiempo": r["elasticidad_media"],
                    "elasticidad_sd": r["elasticidad_sd"],
                    "pct_particiones_positiva": r["pct_positiva"],
                    "n_particiones": r["n_particiones"],
                })
        else:
            print(f"[{ds}] falta {f.name}")

        # ---- 5. Benchmark de JSD  (Revisor 1, C5) -> de jsd_{ds}.csv --------
        f = res / f"jsd_{ds}.csv"
        if f.exists():
            j = pd.read_csv(f)
            # Misma prueba que 06_diagnosticos.py: cada JSD (variable, modo)
            # contra el p95 del benchmark de ESA variable en ESE modo. No se
            # compara un promedio de JSD con un promedio de p95: una variable
            # con p95 alto compensaría a otra con p95 bajo.
            modos = [c[len("benchmark_p95_modo_"):] for c in j.columns
                     if c.startswith("benchmark_p95_modo_")]
            for metodo, g in j.groupby("metodo"):
                jsd = g[[f"jsd_modo_{m}" for m in modos]].to_numpy(float)
                p95 = g[[f"benchmark_p95_modo_{m}" for m in modos]].to_numpy(float)
                ok = np.isfinite(jsd) & np.isfinite(p95)
                jsd, p95 = jsd[ok], p95[ok]
                # p95 = 0 ocurre cuando la variable es constante en ese modo.
                cociente = np.where(p95 > 0, jsd / np.where(p95 > 0, p95, 1.0),
                                    np.where(jsd > 0, np.inf, 0.0))
                filas_jsd.append({
                    "dataset": etq, "metodo": metodo,
                    "n_vars": int(g["variable"].nunique()),
                    "vars_sobre_benchmark": int(g["supera_benchmark"].sum()),
                    "n_pares_var_modo": int(ok.sum()),
                    "pares_sobre_p95": int((jsd > p95).sum()),
                    "cociente_jsd_p95_mediana":
                        float(np.median(cociente)) if cociente.size else np.nan,
                    "jsd_sintetico_medio":
                        float(np.mean(jsd)) if jsd.size else np.nan,
                    "jsd_benchmark_p95_medio":
                        float(np.mean(p95)) if p95.size else np.nan,
                })
        else:
            print(f"[{ds}] falta {f.name} (corre 06_diagnosticos.py)")

    # ---- escribir tablas combinadas -----------------------------------------
    if filas_vtt:
        pd.DataFrame(filas_vtt).round(6).to_csv(out / "tabla_vtt.csv", index=False)
    if filas_elas:
        pd.DataFrame(filas_elas).round(6).to_csv(
            out / "tabla_elasticidades.csv", index=False)
    if filas_jsd:
        pd.DataFrame(filas_jsd).round(5).to_csv(
            out / "jsd_benchmark.csv", index=False)

    print(f"\nEntregables del PDF escritos en: {out}")
    for p in sorted(out.glob("*.csv")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
