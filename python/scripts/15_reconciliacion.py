#!/usr/bin/env python3
"""
Paso 15 — Reconciliación celda por celda: paper publicado vs. pipeline.

Entradas:
    docs/valores_paper.csv     valores publicados, transcritos del PDF
                               (docs/Paper_Aceptado_TRR_CVAE.pdf) con página
    tablas_paper/tabla_NN.csv  salida de 14_tablas_paper.py
    results/corrigendum/...    para las afirmaciones numéricas del texto

Salida:
    docs/reconciliacion_paper.csv   una fila por celda: tabla, dataset, fila,
        columna, valor_paper, valor_pipeline, diferencia, archivo_fuente,
        estado, causa

Estados:
    coincide        mismo valor
    redondeo        difiere en a lo sumo una unidad del último decimal publicado
    procedencia     el número publicado salió de otra versión o de otra regla
                    de agregación (fuga de datos, filtro de convergencia,
                    partición única, columnas no generadas)
    especificacion  cambió el modelo (Swissmetro: spec reducida -> D3 + G1)
    aritmetica      cuenta o cociente mal hecho en el texto
    genuina         diferencia que ninguna de las causas anteriores explica

La causa de cada regla remite a las notas N1–N9 y §§ de
docs/diagnostico_corrigendum.md.

Uso:
    python python/scripts/15_reconciliacion.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import RAIZ, cargar  # noqa: E402

SW, SM, LC = "Swissmetro", "SM-Centro", "LC-Centro"
ESPEC_SW = ("especificacion", "Swissmetro re-especificado: spec reducida publicada -> "
            "spec completa + D3 (CAR_AV) + G1 (costo efectivo) (§6, §10, §11)")

# (tabla, dataset o None, regex de columna o None, estado, causa).
# Se usan solo si la celda NO coincide ni difiere por redondeo; gana la primera.
REGLAS = [
    ("tabla_06", None, None, "aritmetica",
     "paper: N de la muestra completa y sintéticos de una sola clase (SM) o de la "
     "submuestra con auto (SW, N = 9.036); pipeline: media por fold de entrenamiento, "
     "todas las clases balanceadas (V01, V04)"),
    ("tabla_07", SW, None, "especificacion",
     "sintéticos regenerados con D3 + G1; además el JSD publicado incluía columnas "
     "no generadas (N1)"),
    ("tabla_07", None, None, "procedencia",
     "el JSD publicado promediaba columnas que el método no genera (ID, cvae.excluir) (N1)"),
    ("tabla_08", SW, None, *ESPEC_SW),
    ("tabla_08", SM, None, "procedencia",
     "el paper promedia solo los folds marcados como convergidos (class_weights: 11/50); "
     "se reproduce exacto desde results/estimaciones_sm_centro.csv filtrando convergio "
     "(§2, regla única de convergio)"),
    ("tabla_08", LC, None, "procedencia",
     "el paper publicó la Tabla 8 de LC-Centro con la spec SIN ingreso (coincide exacta "
     "con results/corrigendum_lc_sin_ingreso/), mientras la columna ORIG. de la Tabla 11 "
     "se estimó CON ingreso; el corrigendum usa la spec de la Tabla 4, con ingreso (N13)"),
    ("tabla_09", None, None, "procedencia",
     "las cinco celdas publicadas (todos los métodos, no solo el CVAE) salen de un único "
     "modelo por método estimado en la partición (0, 0); el pipeline promedia las 50 (N9)"),
    ("tabla_10", SW, None, *ESPEC_SW),
    ("tabla_10", SM, None, "procedencia",
     "el paper promedia solo los folds convergidos (§2)"),
    ("tabla_10", LC, None, "procedencia",
     "publicada con la spec de LC-Centro SIN ingreso; el corrigendum usa la de la "
     "Tabla 4, con ingreso (N13)"),
    ("tabla_11", SW, None, *ESPEC_SW),
    # ORIG. de SM-Centro y LC-Centro: sin regla. El MLE de muestra completa (LC ya
    # con ingreso, N10) reproduce el publicado; lo que quede es redondeo.
    ("tabla_11", SM, r"^ORIG\.", None, None),
    ("tabla_11", LC, r"^ORIG\.", None, None),
    ("tabla_11", None, None, "procedencia",
     "columnas por método tomadas de la versión sometida (con fuga de datos); no se "
     "reproducen desde los resultados validados ni desde el pipeline corregido"),
]

# Afirmaciones del texto: estado si no coinciden.
REGLAS_TEXTO = {
    "fraccion sintetica %": ("aritmetica",
                             "222 / (222 + 53): sintéticos de un fold sobre reales de la "
                             "muestra completa; por fold es 223,2 / (223,2 + 42,4) (V04)"),
    "N muestra": ("aritmetica", "9.036 es la submuestra con auto disponible "
                  "(CAR_TT > 0); la muestra estimada es 10.710 (V01)"),
    "JSD CVAE / JSD ROS": ("procedencia", "JSD sobre columnas no generadas (N1); en "
                           "Swissmetro además D3 + G1"),
    "coeficientes invertidos exclusivos": ("procedencia",
                                           "el texto sigue la Tabla 11 sometida; sobre "
                                           "50 particiones b_ivt del CVAE se invierte "
                                           "solo en el 30 % (N9)"),
    "b_cost negativo en todos los metodos": ("especificacion", ESPEC_SW[1]),
    "datasets con inversion exclusiva del CVAE": ("especificacion", ESPEC_SW[1]),
    "fuente de las utilidades": ("procedencia", "Salas et al. (2022) no estima "
                                 "Swissmetro (N7)"),
    "datasets donde el CVAE mejora el F1 del original": (
        "especificacion", "con la spec de la Tabla 4 (LC con ingreso, N13) y Swissmetro "
        "re-especificado, el CVAE supera al original también en LC-Centro"),
    "datasets donde el CVAE tiene el mayor F1": (
        "especificacion", "con Swissmetro re-especificado (D3 + G1) el CVAE tiene el F1 "
        "más alto también en Swissmetro, con diferencias menores que la SD entre "
        "particiones (§11)"),
    "variables de la utilidad: jefe de hogar (HH)": (
        "procedencia", "la Tabla 4 declara HH, pero la estimación publicada no lo "
        "incluía: con HH la columna ORIG. publicada no se reproduce (N10); el "
        "corrigendum retira HH de la Tabla 4"),
    "efecto sistematico del CVAE entre particiones": (
        "procedencia", "el texto atribuye al tiempo de viaje del auto un patrón "
        "sistemático que viene de una sola partición; sobre 50 particiones el efecto "
        "sistemático del CVAE está en el tiempo de espera (N9, "
        "docs/parrafo_elasticidades_corrigendum.md)"),
}


def decimales(s: str) -> int:
    m = re.search(r"\.(\d+)$", s.strip())
    return len(m.group(1)) if m else 0


def a_num(s):
    try:
        return float(str(s).replace(",", "").replace("+", ""))
    except ValueError:
        return np.nan


def largo_pipeline(cfg) -> pd.DataFrame:
    """tablas_paper/*.csv en formato largo (dataset, fila, columna, valor)."""
    P = cfg.d["paper"]
    claves = {"tabla_02": "Travel Mode", "tabla_08": "Method",
              "tabla_09": "Method", "tabla_11": "Parameter"}
    partes = []
    for tabla in ["tabla_02", "tabla_06", "tabla_07", "tabla_08",
                  "tabla_09", "tabla_10", "tabla_11"]:
        ruta = RAIZ / P["destino"] / f"{tabla}.csv"
        t = pd.read_csv(ruta, comment="#", dtype=str)
        if "Dataset" not in t:
            t["Dataset"] = P["datasets"][P["tabla_09"]["dataset"]]
        fila = claves.get(tabla)
        t["fila"] = t[fila] if fila else ""
        valores = [c for c in t.columns if c not in ("Dataset", "fila", fila,
                                                      "Minority class")]
        l = t.melt(id_vars=["Dataset", "fila"], value_vars=valores,
                   var_name="columna", value_name="valor_pipeline")
        l.insert(0, "tabla", tabla)
        l["archivo_fuente"] = ruta.relative_to(RAIZ).as_posix()
        partes.append(l.rename(columns={"Dataset": "dataset"}))
    return pd.concat(partes, ignore_index=True)


def afirmaciones_texto(cfg) -> pd.DataFrame:
    """Valor del pipeline para cada afirmación numérica del texto."""
    R, P = cfg.rutas.results, cfg.d["paper"]
    rel = lambda p: Path(p).relative_to(RAIZ).as_posix()
    filas = []

    pc = pd.read_csv(RAIZ / P["destino"] / "tabla_06_por_clase.csv", comment="#")
    v = pc[(pc.Dataset == SM) & (pc["Travel Mode"] == P["modos"]["sm_centro"][1])]
    filas.append((SM, "Car Driver", "fraccion sintetica %",
                  f"{v['Synthetic share %'].iloc[0]:.1f}",
                  f"{P['destino']}/tabla_06_por_clase.csv"))
    filas.append((SW, "", "N muestra", str(len(pd.read_csv(cfg.ruta_original("swissmetro")))),
                  rel(cfg.ruta_original("swissmetro"))))
    for ds, etq in (("sm_centro", SM), ("swissmetro", SW)):
        j = pd.read_csv(R / f"jsd_{ds}.csv").groupby("metodo").jsd_modo_medio.mean()
        filas.append((etq, "", "JSD CVAE / JSD ROS", f"{j['CVAE'] / j['ROS']:.0f}",
                      rel(R / f"jsd_{ds}.csv")))

    base = ["original", "ROS", "SMOTE", "class_weights"]
    exclusivos = {}
    for ds, etq in P["datasets"].items():
        s = pd.read_csv(R / f"tabla_signos_{ds}.csv")
        inv = s[s.invertido_en_media]
        cvae = set(inv[inv.metodo == "CVAE"].coeficiente)
        otros = set(inv[inv.metodo.isin(base)].coeficiente)
        exclusivos[ds] = sorted(cvae - otros)
        if ds != "swissmetro":
            filas.append((etq, "CVAE", "coeficientes invertidos exclusivos",
                          "; ".join(exclusivos[ds]), rel(R / f"tabla_signos_{ds}.csv")))
    s = pd.read_csv(R / "tabla_signos_swissmetro.csv")
    neg = not s[s.coeficiente == "b_cost"].invertido_en_media.any()
    filas.append((SW, "todos", "b_cost negativo en todos los metodos",
                  "si" if neg else "no", rel(R / "tabla_signos_swissmetro.csv")))
    filas.append(("todos", "CVAE", "datasets con inversion exclusiva del CVAE",
                  str(sum(bool(v) for v in exclusivos.values())),
                  f"{rel(R)}/tabla_signos_*.csv"))
    # HH (jefe de hogar, columna JEFH) en la utilidad de LC-Centro, según config.
    usa_hh = any("JEFH" in a["atributos"].values()
                 for a in cfg.dataset("lc_centro")["modelo"]["alternativas"])
    filas.append((LC, "", "variables de la utilidad: jefe de hogar (HH)",
                  "si" if usa_hh else "no", "config.yaml (datasets.lc_centro.modelo)"))
    # Covariables cuya elasticidad PROPIA es positiva bajo CVAE en todas las
    # particiones (SM-Centro): el efecto sistemático entre particiones.
    ds9 = P["tabla_09"]["dataset"]
    er = pd.read_csv(R / f"efectos_marginales_resumen_{ds9}.csv")
    # Solo covariables cuyo coeficiente tiene signo esperado negativo: una
    # elasticidad propia positiva en ellas es una reversión.
    esperados = cfg.dataset(ds9)["modelo"].get("signos_esperados", {})
    attr = {a["id"]: {col for b, col in a["atributos"].items() if esperados.get(b) == -1}
            for a in cfg.dataset(ds9)["modelo"]["alternativas"]}
    propia = er[er.apply(lambda r: r.covariable in attr[r.alternativa], axis=1)]
    sist = propia[(propia.metodo == "CVAE") & (propia.pct_positiva == 1)]
    filas.append((P["datasets"][ds9], "CVAE", "efecto sistematico del CVAE entre particiones",
                  "; ".join(sorted(sist.covariable)),
                  rel(R / f"efectos_marginales_resumen_{ds9}.csv")))
    # Desempeño predictivo del CVAE frente al original y al resto (Tabla 8).
    t8 = pd.read_csv(RAIZ / P["destino"] / "tabla_08.csv", comment="#")
    f1 = t8.pivot(index="Dataset", columns="Method", values="F1 Macro")
    cvae, orig = P["metodos"]["CVAE"], P["metodos"]["original"]
    mejora = [d for d in P["datasets"].values() if f1.loc[d, cvae] > f1.loc[d, orig]]
    mejor = [d for d in P["datasets"].values() if f1.loc[d].idxmax() == cvae]
    filas.append(("todos", "CVAE", "datasets donde el CVAE mejora el F1 del original",
                  str(len(mejora)), f"{P['destino']}/tabla_08.csv"))
    filas.append(("todos", "CVAE", "datasets donde el CVAE tiene el mayor F1",
                  "; ".join(mejor), f"{P['destino']}/tabla_08.csv"))
    filas.append((SW, "", "fuente de las utilidades",
                  "convención Biogeme (Bierlaire et al., 2001)",
                  "docs/diagnostico_corrigendum.md §10"))
    t = pd.DataFrame(filas, columns=["dataset", "fila", "columna", "valor_pipeline",
                                     "archivo_fuente"])
    t.insert(0, "tabla", "texto")
    return t


def clasificar(r) -> tuple[str, str]:
    vp, vq = str(r.valor_paper), str(r.valor_pipeline)
    if r.valor_pipeline is None or pd.isna(r.valor_pipeline):
        return "sin_equivalente", "la celda no tiene equivalente en el pipeline"
    xp, xq = a_num(vp), a_num(vq)
    numerico = not (np.isnan(xp) or np.isnan(xq))
    if (numerico and xp == xq) or (not numerico and vp.strip() == vq.strip()):
        return "coincide", ""
    # Una causa conocida (otra versión, otra regla de agregación, otro modelo)
    # gana sobre el redondeo: si el número salió de otra fuente, que difiera
    # en una unidad no lo convierte en un problema de redondeo.
    if r.tabla == "texto":
        return REGLAS_TEXTO.get(r.columna, ("genuina", "sin causa identificada"))
    for tabla, ds, col, estado, causa in REGLAS:
        if tabla == r.tabla and (ds is None or ds == r.dataset) and \
                (col is None or re.search(col, r.columna)):
            if estado is None:      # regla de exclusión: sin causa externa
                break
            return estado, causa
    if numerico and abs(xq - xp) <= 10 ** -decimales(vp) + 1e-12:
        return "redondeo", "diferencia de una unidad en el último decimal"
    return "genuina", "sin causa identificada"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--valores", default=str(RAIZ / "docs" / "valores_paper.csv"))
    ap.add_argument("--destino", default=str(RAIZ / "docs" / "reconciliacion_paper.csv"))
    args = ap.parse_args()

    cfg = cargar()
    paper = pd.read_csv(args.valores, dtype=str).fillna("")
    pipe = pd.concat([largo_pipeline(cfg), afirmaciones_texto(cfg)], ignore_index=True)
    pipe["fila"] = pipe["fila"].fillna("")
    t = paper.merge(pipe, on=["tabla", "dataset", "fila", "columna"], how="left")
    faltan = t[t.valor_pipeline.isna()]
    if len(faltan):
        print(f"AVISO: {len(faltan)} celdas del paper sin equivalente:\n"
              f"{faltan[['tabla', 'dataset', 'fila', 'columna']].to_string(index=False)}")
    t[["estado", "causa"]] = t.apply(lambda r: pd.Series(clasificar(r)), axis=1)
    t["diferencia"] = [
        (f"{a_num(q) - a_num(p):+.4g}" if not (np.isnan(a_num(p)) or np.isnan(a_num(q))) else "")
        for p, q in zip(t.valor_paper, t.valor_pipeline)]
    cols = ["tabla", "dataset", "fila", "columna", "valor_paper", "valor_pipeline",
            "diferencia", "archivo_fuente", "estado", "causa", "ubicacion"]
    destino = Path(args.destino)
    t[cols].to_csv(destino, index=False)
    print(f"{len(t)} celdas -> {destino}\n")
    print(pd.crosstab(t.tabla, t.estado, margins=True).to_string())


if __name__ == "__main__":
    main()
