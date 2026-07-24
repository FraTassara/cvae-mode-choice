#!/usr/bin/env python3
"""
Paso 6 — Diagnósticos de calidad de los datos sintéticos.

Cubre cuatro puntos del plan de revisores que estaban implementados en el
paquete pero que ningún script invocaba:

  §2.3(a)  JSD por variable Y POR MODO, real vs. sintético
  §2.3(b)  Gaps de atributos (elegido − no elegido), real vs. sintético
  §2.5     Benchmark interno de JSD (dos mitades aleatorias de los datos reales)
  §2.6     Ratio de sobremuestreo por clase y por método

Lee lo que ya dejaron los pasos 1 y 2; no reejecuta nada ni reentrena el CVAE.

Uso:
    python python/scripts/06_diagnosticos.py --dataset sm_centro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae import diagnostics  # noqa: E402
from cvae.config_repo import cargar  # noqa: E402


def _columnas_por_modo(d: dict) -> dict[int, list[str]]:
    """
    Construye {id_modo: [col_1, col_2, ...]} en orden CONSISTENTE entre modos,
    a partir del bloque `modelo.alternativas` de config.yaml.

    Solo usa los betas que aparecen en TODAS las alternativas (genéricos): son
    los únicos comparables posición a posición. En datasets con coeficientes
    específicos por modo (p.ej. Whalen) la intersección puede quedar vacía y
    el análisis §2.3(b) se omite.
    """
    alts = d["modelo"]["alternativas"]
    comunes = set(alts[0]["atributos"])
    for a in alts[1:]:
        comunes &= set(a["atributos"])
    if not comunes:
        return {}
    orden = sorted(comunes)
    return {int(a["id"]): [a["atributos"][b] for b in orden] for a in alts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bins", type=int, default=30,
                    help="bins de los histogramas para el JSD")
    ap.add_argument("--repeticiones-benchmark", type=int, default=50)
    args = ap.parse_args()

    cfg = cargar()
    ds = args.dataset
    d = cfg.dataset(ds)
    choice = d["choice_col"]

    original = pd.read_csv(cfg.ruta_original(ds))
    ignorar = {"row_id", "repeticion", "fold", "is_synthetic"}
    cols_datos = [c for c in original.columns if c not in ignorar]

    # ------------------------------------------------------------------ #
    #  §2.5 — Benchmark interno: JSD entre dos mitades de los datos reales.
    #  Es la escala de referencia; se calcula una sola vez por dataset.
    # ------------------------------------------------------------------ #
    print(f"\n[{ds}] §2.5 Benchmark interno de JSD "
          f"({args.repeticiones_benchmark} repeticiones)...")
    # El benchmark se calcula POR MODO, sobre las filas reales de ese modo.
    # Si se calculara sobre el dataset completo, se compararía la variabilidad
    # de una clase minoritaria (pocas filas, mucha varianza) contra la de la
    # muestra entera (muchas filas, poca varianza), y toda clase pequeña
    # parecería mal generada.
    bench_por_modo = {}
    for modo in sorted(original[choice].unique()):
        sub = original.loc[original[choice] == modo, cols_datos]
        if len(sub) < 10:
            print(f"  modo {modo}: solo {len(sub)} filas reales, "
                  "benchmark poco fiable.")
        bench_por_modo[int(modo)] = diagnostics.jsd_internal_benchmark(
            sub, target_col=choice, n_repeats=args.repeticiones_benchmark,
            bins=args.bins, seed=cfg.semillas["folds"])

    # ------------------------------------------------------------------ #
    #  Por método: JSD, ratio de sobremuestreo y gaps de atributos.
    # ------------------------------------------------------------------ #
    jsd_todos, ratios_todos, gaps_todos = [], [], []
    mapa_cols = _columnas_por_modo(d)

    if mapa_cols:
        gaps_real = diagnostics.attribute_gaps(original, choice, mapa_cols)
        gaps_real["fuente"] = "real"
        gaps_todos.append(gaps_real)
    else:
        print("  AVISO: los coeficientes no son genéricos entre modos; "
              "se omite §2.3(b).")

    for metodo in cfg.metodos:
        ruta = cfg.ruta_train(ds, metodo)
        if not ruta.exists():
            print(f"  {metodo}: sin datos ({ruta.name}), se salta.")
            continue

        train = pd.read_csv(ruta, compression="gzip")
        synth = train[train.is_synthetic == 1]
        if synth.empty:
            print(f"  {metodo}: no genera sintéticos (línea base), se salta.")
            continue

        # Una sola partición basta para caracterizar la distribución generada:
        # las 50 son el mismo procedimiento con distinta semilla.
        synth_0 = synth[(synth.repeticion == 0) & (synth.fold == 0)]
        if synth_0.empty:
            synth_0 = synth

        print(f"  {metodo}: {len(synth_0)} filas sintéticas (partición 0)")

        # --- §2.3(a) JSD por variable y por modo -----------------------
        j = diagnostics.jsd_by_variable(
            original[cols_datos], synth_0[cols_datos],
            target_col=choice, bins=args.bins)
        j.insert(0, "metodo", metodo)
        jsd_todos.append(j.reset_index())

        # --- §2.6 Ratio de sobremuestreo -------------------------------
        r = diagnostics.oversampling_report(original, synth_0, choice, metodo)
        ratios_todos.append(r.reset_index())

        # --- §2.3(b) Gaps de atributos ---------------------------------
        if mapa_cols:
            g = diagnostics.attribute_gaps(synth_0, choice, mapa_cols)
            g["fuente"] = metodo
            gaps_todos.append(g)

    if not jsd_todos:
        raise SystemExit(
            "No había conjuntos con filas sintéticas. Corre primero:\n"
            f"  python python/scripts/02_sinteticos.py --dataset {ds}")

    # ------------------------------------------------------------------ #
    #  Salidas
    # ------------------------------------------------------------------ #
    res = cfg.rutas.results

    jsd_tab = pd.concat(jsd_todos, ignore_index=True)

    # Cada JSD por modo se compara contra el benchmark DE ESE MODO, para que
    # los tamaños muestrales sean comparables. Se cuenta una variable como
    # "distorsionada" si supera el p95 del benchmark en algún modo.
    cols_modo = [c for c in jsd_tab.columns if c.startswith("jsd_modo_")]
    excesos = pd.DataFrame(index=jsd_tab.index)
    for c in cols_modo:
        modo = int(c.replace("jsd_modo_", ""))
        b = bench_por_modo.get(modo)
        if b is None:
            continue
        p95 = jsd_tab["variable"].map(b["jsd_benchmark_p95"])
        jsd_tab[f"benchmark_p95_modo_{modo}"] = p95
        excesos[c] = jsd_tab[c] > p95

    jsd_tab["jsd_modo_medio"] = jsd_tab[cols_modo].mean(axis=1) if cols_modo else float("nan")
    jsd_tab["supera_benchmark"] = excesos.any(axis=1) if len(excesos.columns) else False
    jsd_tab["n_modos_sobre_benchmark"] = (excesos.sum(axis=1)
                                          if len(excesos.columns) else 0)
    jsd_tab.round(5).to_csv(res / f"jsd_{ds}.csv", index=False)

    ratio_tab = pd.concat(ratios_todos, ignore_index=True)
    ratio_tab.to_csv(res / f"sobremuestreo_{ds}.csv", index=False)

    print(f"\n--- §2.6 Ratio de sobremuestreo ---")
    print(ratio_tab.to_string(index=False))

    print(f"\n--- §2.5 y §2.3(a) JSD por modo vs. benchmark del mismo modo ---")
    resumen = (jsd_tab.groupby("metodo")
               .agg(jsd_modo_medio=("jsd_modo_medio", "mean"),
                    jsd_modo_max=("jsd_modo_medio", "max"),
                    vars_sobre_benchmark=("supera_benchmark", "sum"),
                    n_vars=("supera_benchmark", "size")))
    print(resumen.round(4).to_string())
    print("\nUna variable 'sobre benchmark' se desvía más de lo que se desvían")
    print("dos mitades aleatorias de los datos reales de ese mismo modo.")
    print("(detalle por variable y por modo en el CSV)")

    if gaps_todos:
        gaps_tab = pd.concat(gaps_todos, ignore_index=True)
        gaps_tab.round(4).to_csv(res / f"gaps_atributos_{ds}.csv", index=False)
        pivote = gaps_tab.pivot_table(index=["modo_elegido", "col"],
                                      columns="fuente", values="gap")
        print(f"\n--- §2.3(b) Gaps de atributos (elegido − no elegido) ---")
        print(pivote.round(3).to_string())
        print("\nSi el gap real y el sintético difieren en signo o magnitud,")
        print("ahí está el origen de las reversiones de signo del MNL.")

    print(f"\nArchivos escritos en {res}/:")
    print(f"  jsd_{ds}.csv, sobremuestreo_{ds}.csv"
          + (f", gaps_atributos_{ds}.csv" if gaps_todos else ""))


if __name__ == "__main__":
    main()
