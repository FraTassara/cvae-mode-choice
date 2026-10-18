#!/usr/bin/env python3
"""
Paso 14 — Tablas del paper (2, 6, 7, 8, 9, 10 y 11) desde las salidas del pipeline.

NO estima ni recalcula modelos: lee lo que dejaron los pasos 1–10 en
`proyecto.rutas.results` y lo reescribe con el formato de las tablas del paper
(docs/Paper_Aceptado_TRR_CVAE.pdf). Etiquetas y orden salen del bloque
`paper:` de config.yaml; ningún número se escribe a mano.

Cada CSV lleva en el pie (líneas que empiezan con '#') el commit de git, el
hash del MANIFEST.json y los archivos de origen. Para leerlos:
    pd.read_csv(ruta, comment="#")

Entradas (en results/):
    {ds}_original.csv (data/processed)    -> Tabla 2
    sobremuestreo_{ds}.csv                 -> Tabla 6   [06_diagnosticos]
    jsd_{ds}.csv                           -> Tabla 7   [06_diagnosticos]
    tabla_predictiva_{ds}.csv              -> Tabla 8   [04_reporte]
    efectos_marginales_resumen_{ds}.csv    -> Tabla 9   [08_efectos_marginales]
    sensibilidad_elipses_{ds}.csv          -> Tabla 10  [10_sensibilidad]
    estimaciones_completa_{ds}.csv         -> Tabla 11, ORIG.  [03 --muestra-completa]
    tabla_coeficientes_{ds}.csv            -> Tabla 11, métodos [04_reporte]

Uso:
    python python/scripts/14_tablas_paper.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import RAIZ, cargar  # noqa: E402


# --------------------------------------------------------------------------- #
#  Formato de números como en el paper
# --------------------------------------------------------------------------- #
def f_dec(x, d=3) -> str:
    return "" if pd.isna(x) else f"{x:.{d}f}"


def f_signo(x, d=3) -> str:
    """+0.014 / -0.180, como en las Tablas 9 y 11."""
    return "" if pd.isna(x) else f"{x:+.{d}f}"


def f_n(x) -> str:
    """9,036; las medias por fold no enteras llevan un decimal (340.8)."""
    if pd.isna(x):
        return ""
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.1f}"


# --------------------------------------------------------------------------- #
#  Procedencia
# --------------------------------------------------------------------------- #
def procedencia(cfg) -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=RAIZ, capture_output=True,
                              text=True, check=True).stdout.strip()

    commit = git("rev-parse", "HEAD")
    sucio = git("status", "--porcelain", "--untracked-files=no")
    manifiesto = cfg.rutas.processed / "MANIFEST.json"
    m = json.loads(manifiesto.read_text())
    return {
        "git_commit": commit + (" (con cambios sin commitear)" if sucio else ""),
        "manifest_sha256": hashlib.sha256(manifiesto.read_bytes()).hexdigest()[:12],
        "config_hash": m["config_hash"],
    }


def escribir(tabla: pd.DataFrame, nombre: str, fuentes: list[Path], prov: dict,
             cfg, notas: list[str] = ()) -> None:
    destino = RAIZ / cfg.d["paper"]["destino"] / f"{nombre}.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino, index=False)
    with open(destino, "a", encoding="utf-8", newline="") as f:
        for nota in notas:
            f.write(f"# nota: {nota}\n")
        for p in fuentes:
            f.write(f"# fuente: {p.relative_to(RAIZ).as_posix()}\n")
        for k, v in prov.items():
            f.write(f"# {k}: {v}\n")
        f.write("# generado por: python/scripts/14_tablas_paper.py\n")
    print(f"\n--- {nombre} ---\n{tabla.to_string(index=False)}\n-> {destino.relative_to(RAIZ)}")


def leer(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise SystemExit(f"Falta {p.relative_to(RAIZ)}. Corre el paso que lo produce.")
    return pd.read_csv(p)


# --------------------------------------------------------------------------- #
#  Tablas
# --------------------------------------------------------------------------- #
def tabla_02(cfg, P, prov):
    filas, fuentes = [], []
    for ds, etq in P["datasets"].items():
        ruta = cfg.ruta_original(ds)
        fuentes.append(ruta)
        y = leer(ruta)[cfg.dataset(ds)["choice_col"]]
        n = y.value_counts()
        for modo, nombre in P["modos"][ds].items():
            filas.append({"Dataset": etq, "Travel Mode": nombre,
                          "Frequency": f_n(n.get(modo, 0)),
                          "Share %": f_dec(100 * n.get(modo, 0) / len(y), 2)})
    escribir(pd.DataFrame(filas), "tabla_02", fuentes, prov, cfg)


def tabla_06(cfg, P, prov):
    filas, por_clase, fuentes = [], [], []
    for ds, etq in P["datasets"].items():
        ruta = cfg.rutas.results / f"sobremuestreo_{ds}.csv"
        fuentes.append(ruta)
        s = leer(ruta)
        choice = cfg.dataset(ds)["choice_col"]
        # ROS, SMOTE y CVAE balancean al mismo objetivo: los conteos coinciden.
        conteos = s.groupby("metodo")[["n_real_media", "n_sintetico_media"]].sum()
        if conteos.nunique().max() > 1:
            raise SystemExit(f"[{ds}] los métodos no generan el mismo número de "
                             f"sintéticos:\n{conteos}")
        g = s[s.metodo == P["tabla_06_metodo"]]
        real, sint = g.n_real_media.sum(), g.n_sintetico_media.sum()
        filas.append({"Dataset": etq, "Original (N)": f_n(real),
                      "Synthetic (N)": f_n(sint), "Ratio": f_dec(sint / real, 2)})
        for _, r in g.iterrows():
            por_clase.append({
                "Dataset": etq, "Travel Mode": P["modos"][ds][int(r[choice])],
                "Real (N)": f_n(r.n_real_media),
                "Synthetic (N)": f_n(r.n_sintetico_media),
                "Synthetic share %": f_dec(100 * r.frac_sintetica_media, 1)})
    nota = ("N = media por fold de entrenamiento (50 particiones); "
            "balanceo a la clase mayoritaria dentro de cada fold")
    escribir(pd.DataFrame(filas), "tabla_06", fuentes, prov, cfg, [nota])
    escribir(pd.DataFrame(por_clase), "tabla_06_por_clase", fuentes, prov, cfg, [nota])


def tabla_07(cfg, P, prov):
    filas, fuentes = [], []
    for ds, etq in P["datasets"].items():
        ruta = cfg.rutas.results / f"jsd_{ds}.csv"
        fuentes.append(ruta)
        j = leer(ruta).groupby("metodo").jsd_modo_medio.mean()
        fila = {"Dataset": etq}
        for m in P["tabla_07_metodos"]:
            fila[f"JSD {m}"] = f_dec(j.get(m), 3)
        filas.append(fila)
    nota = ("media de jsd_modo_medio sobre las variables que el método genera "
            "(sin id_col ni cvae.excluir), partición (0, 0)")
    escribir(pd.DataFrame(filas), "tabla_07", fuentes, prov, cfg, [nota])


def tabla_08(cfg, P, prov):
    filas, fuentes = [], []
    for ds, etq in P["datasets"].items():
        ruta = cfg.rutas.results / f"tabla_predictiva_{ds}.csv"
        fuentes += [ruta, cfg.ruta_original(ds)]
        tp = leer(ruta).set_index("metodo")
        # Clase minoritaria: la de menor frecuencia en los datos reales.
        y = leer(cfg.ruta_original(ds))[cfg.dataset(ds)["choice_col"]]
        minoritaria = int(y.value_counts().idxmin())
        for m, etq_m in P["metodos"].items():
            r = tp.loc[m]
            filas.append({"Dataset": etq, "Method": etq_m,
                          "Accuracy": f_dec(r.accuracy_media),
                          "F1 Macro": f_dec(r.f1_macro_media),
                          "Recall Minority": f_dec(r[f"recall_{minoritaria}_media"]),
                          "Minority class": P["modos"][ds][minoritaria]})
    escribir(pd.DataFrame(filas), "tabla_08", fuentes, prov, cfg)


def tabla_09(cfg, P, prov):
    t9 = P["tabla_09"]
    ds = t9["dataset"]
    alt = next(a for a in cfg.dataset(ds)["modelo"]["alternativas"]
               if a["id"] == t9["alternativa"])
    covariable = alt["atributos"][t9["coeficiente"]]
    ruta = cfg.rutas.results / f"efectos_marginales_resumen_{ds}.csv"
    er = leer(ruta)
    er = er[(er.covariable == covariable) & (er.alternativa == t9["alternativa"])]
    er = er.set_index("metodo")
    filas = []
    for m, etq_m in P["metodos"].items():
        r = er.loc[m]
        filas.append({"Method": etq_m,
                      "Own-elasticity of travel time": f_signo(r.elasticidad_media),
                      "S.D. across partitions": f_dec(r.elasticidad_sd),
                      "% partitions positive": f_dec(100 * r.pct_positiva, 0),
                      "N partitions": int(r.n_particiones)})
    nota = (f"{P['datasets'][ds]}, alternativa {P['modos'][ds][t9['alternativa']]}, "
            f"covariable {covariable}: media y SD sobre las particiones (N9)")
    escribir(pd.DataFrame(filas), "tabla_09", [ruta], prov, cfg, [nota])


def tabla_10(cfg, P, prov):
    filas, fuentes = [], []
    for ds, etq in P["datasets"].items():
        ruta = cfg.rutas.results / f"sensibilidad_elipses_{ds}.csv"
        fuentes.append(ruta)
        s = leer(ruta).set_index("confianza").f1_macro_media.sort_index()
        fila = {"Dataset": etq}
        fila.update({f"{c:.2f}": f_dec(v) for c, v in s.items()})
        filas.append(fila)
    escribir(pd.DataFrame(filas), "tabla_10", fuentes, prov, cfg)


def tabla_11(cfg, P, prov):
    filas, fuentes = [], []
    metodos_cv = P["tabla_11_metodos"]
    for ds, etq in P["datasets"].items():
        r_comp = cfg.ruta_estimaciones_completa(ds)
        r_cv = cfg.rutas.results / f"tabla_coeficientes_{ds}.csv"
        fuentes += [r_comp, r_cv]
        comp = leer(r_comp).iloc[0]
        cv = leer(r_cv).set_index(["metodo", "coeficiente"])
        etiquetas = P["coeficientes"][ds]
        sin_etiqueta = sorted(set(cv.index.get_level_values(1))
                              - set(etiquetas) - {cfg.dataset(ds)["modelo"]["asc_fija"]})
        if sin_etiqueta:
            raise SystemExit(f"[{ds}] coeficientes sin etiqueta en paper.coeficientes: "
                             f"{sin_etiqueta}")
        for coef, etq_c in etiquetas.items():
            fila = {"Dataset": etq, "Parameter": etq_c,
                    "ORIG. beta": f_signo(comp[f"beta_{coef}"]),
                    "ORIG. S.E.": f_dec(comp[f"se_clasico_{coef}"])}
            for m in metodos_cv:
                r = cv.loc[(m, coef)]
                fila[f"{P['metodos_cortos'][m]} beta"] = f_signo(r.media)
                fila[f"{P['metodos_cortos'][m]} S.D.CV"] = f_dec(r.sd_entre_particiones)
            filas.append(fila)
    notas = ["ORIG.: MLE sobre el dataset completo (03_estimar.R --muestra-completa), "
             "S.E. clásico",
             "S.D.CV: desviación de beta entre las 50 particiones",
             "SM-Centro: b_mi_* por cada 10.000 de ILM (modelo.escalas)"]
    escribir(pd.DataFrame(filas), "tabla_11", fuentes, prov, cfg, notas)


def main() -> None:
    cfg = cargar()
    cfg.verificar_manifiesto()
    P = cfg.d["paper"]
    prov = procedencia(cfg)
    for fn in (tabla_02, tabla_06, tabla_07, tabla_08, tabla_09, tabla_10, tabla_11):
        fn(cfg, P, prov)


if __name__ == "__main__":
    main()
