#!/usr/bin/env python3
"""
Experimento: CVAE base vs. CVAE con las dos mejoras, en UN dataset.

Compara, sobre las mismas observaciones reales:
  A) CVAE base:        n_z = 2, KL global (lo que usa el pipeline actual).
  B) CVAE mejorado:    n_z alto (--n-z) + KL relajado por clase (--kl-min).

Mide el efecto de las mejoras con métricas independientes de la dimensión
latente:
  - JSD por modo (calidad de generación), con foco en la clase minoritaria.
  - Mahalanobis²: chequeo de que el muestreo llena bien el elipsoide en n_z dims.
Y produce dos figuras del espacio latente (base en 2D directo; mejorado en PCA).

NO toca el pipeline: importa de `cvae` lo reutilizable y usa el modelo
experimental para la variante B.

Uso:
    python experimental/exp_comparar.py --dataset sm_centro
    python experimental/exp_comparar.py --dataset sm_centro --n-z 8 --kl-min 0.3
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# La consola de Windows (cp1252) no imprime algunos caracteres (Δ, tildes en
# ciertos contextos). Forzar UTF-8 en la salida evita UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats

# raíz del repo en el path, para importar `cvae` y `experimental`
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "python"))
sys.path.insert(0, str(RAIZ))

from cvae.config_repo import cargar  # noqa: E402


def _jsd_minoritaria(original, synth, choice, clase_min, cols):
    """JSD medio entre real y sintético, solo filas de la clase minoritaria."""
    from cvae import diagnostics
    r = original[original[choice] == clase_min]
    s = synth[synth[choice] == clase_min] if choice in synth else synth
    if len(r) < 5 or len(s) < 5:
        return float("nan")
    vals = []
    for c in cols:
        if c in r and c in s:
            vals.append(diagnostics.jsd(r[c].to_numpy(), s[c].to_numpy()))
    return float(np.mean(vals)) if vals else float("nan")


def _chequeo_mahalanobis(mu, y, ellipses):
    """
    Fracción de puntos reales dentro de la elipse por clase, comparada con el
    nivel de confianza teórico. Si el muestreo/ajuste es correcto en n_z dims,
    deben coincidir aproximadamente.
    """
    filas = []
    for cls, el in ellipses.items():
        m = mu[np.asarray(y) == cls]
        if len(m) < 5:
            continue
        d2 = el.mahalanobis_sq(m)
        dentro = float(np.mean(d2 <= el.threshold))
        filas.append({"clase": int(cls), "n": len(m),
                      "dentro_empirico": round(dentro, 3),
                      "confianza_teorica": el.confidence})
    return pd.DataFrame(filas)


def _entrenar_y_muestrear(build_fn, dataset, mcfg, gcfg, dcfg, **build_kw):
    """Entrena un CVAE (base o experimental) y devuelve (synth_df, mu, y)."""
    import tensorflow as tf
    from tensorflow.keras import backend as K
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    K.clear_session()

    from cvae.generate import generate

    modelo = build_fn(dataset.n_x, dataset.n_y, mcfg, **build_kw)
    modelo.fit(dataset.X_scaled, dataset.y_cat,
               validation_data=dataset.validation_data, verbose=0)
    synth, _ = generate(modelo, dataset, gcfg, dcfg)
    mu, _ = modelo.encode(dataset.X_scaled, dataset.y_cat)
    y_int = np.argmax(dataset.y_cat, axis=1)
    return synth, mu, y_int


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="sm_centro")
    ap.add_argument("--n-z", type=int, default=8,
                    help="dimensión latente del modelo mejorado")
    ap.add_argument("--kl-min", type=float, default=0.3,
                    help="peso KL de la clase minoritaria (1=sin relajar)")
    args = ap.parse_args()

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    from cvae.data import prepare_from_frame
    from cvae.latent import fit_ellipses
    from cvae.model import build_cvae
    from cvae.pipeline import configs_desde_yaml
    from experimental.model_class_aware import (build_cvae_class_aware,
                                                pesos_kl_por_clase)
    from experimental import plots_pca
    from cvae import plots as plots_base

    cfg = cargar()
    ds = args.dataset
    d = cfg.dataset(ds)
    choice = d["choice_col"]

    origen = cfg.ruta_original(ds)
    datos = pd.read_csv(origen)
    columnas = [c for c in datos.columns if c != "row_id"]

    dcfg, mcfg_base, gcfg = configs_desde_yaml(cfg, ds, cfg.semillas["cvae_muestreo"])
    dataset = prepare_from_frame(datos[columnas], dcfg)

    # Clase minoritaria: índice interno y etiqueta real en los datos.
    y_int_all = np.argmax(dataset.y_cat, axis=1)
    clase_min_int = int(pd.Series(y_int_all).value_counts().idxmin())
    # label_map: {etiqueta_real: índice_interno}; invertimos para recuperar la real.
    inv_label = {v: k for k, v in dataset.label_map.items()}
    clase_min_real = inv_label[clase_min_int]
    ignorar = {"row_id", "is_synthetic", "repeticion", "fold", "peso"}
    cols_datos = [c for c in datos.columns if c not in ignorar and c != choice]

    print(f"[{ds}] clase minoritaria: interna={clase_min_int}, "
          f"real={clase_min_real}")
    print(f"  A) base:     n_z={mcfg_base.n_z}, KL global")
    print(f"  B) mejorado: n_z={args.n_z}, KL minoritaria x{args.kl_min}\n")

    resultados = {}
    figuras = {}

    # ---------- A) MODELO BASE ----------
    print("Entrenando modelo base...")
    synth_a, mu_a, y_a = _entrenar_y_muestrear(
        build_cvae, dataset, mcfg_base, gcfg, dcfg)
    ell_a = fit_ellipses(mu_a, y_a, confidence=gcfg.confidence)
    resultados["base"] = {
        "jsd_min": _jsd_minoritaria(datos, synth_a, choice,
                                    clase_min_real,
                                    cols_datos),
        "n_z": mcfg_base.n_z,
    }
    figuras["base"] = plots_base.latent_scatter(
        mu_a, y_a, ellipses=ell_a,
        title=f"{ds} base (n_z={mcfg_base.n_z})")

    # ---------- B) MODELO MEJORADO ----------
    print("Entrenando modelo mejorado (n_z alto + KL por clase)...")
    from dataclasses import replace
    mcfg_exp = replace(mcfg_base, n_z=args.n_z)
    pesos = pesos_kl_por_clase(y_int_all, args.kl_min)
    print(f"  pesos KL por clase: { {k: round(v,2) for k,v in pesos.items()} }")

    synth_b, mu_b, y_b = _entrenar_y_muestrear(
        build_cvae_class_aware, dataset, mcfg_exp, gcfg, dcfg,
        peso_por_clase=pesos)
    ell_b = fit_ellipses(mu_b, y_b, confidence=gcfg.confidence)
    resultados["mejorado"] = {
        "jsd_min": _jsd_minoritaria(datos, synth_b, choice,
                                    clase_min_real,
                                    cols_datos),
        "n_z": args.n_z,
    }
    figuras["mejorado"] = plots_pca.latent_scatter_pca(
        mu_b, y_b, ellipses=ell_b,
        title=f"{ds} mejorado (n_z={args.n_z}, PCA)")

    # ---------- Chequeo Mahalanobis (muestreo correcto en n_z dims) ----------
    print("\n--- Chequeo de cobertura de elipse (Mahalanobis vs. teórico) ---")
    print("modelo mejorado:")
    print(_chequeo_mahalanobis(mu_b, y_b, ell_b).to_string(index=False))

    # ---------- Salidas ----------
    dest = cfg.rutas.results / "figuras"
    dest.mkdir(parents=True, exist_ok=True)
    figuras["base"].savefig(dest / f"exp_latente_base_{ds}.png",
                            dpi=140, bbox_inches="tight")
    figuras["mejorado"].savefig(dest / f"exp_latente_mejorado_{ds}.png",
                                dpi=140, bbox_inches="tight")

    print("\n--- JSD de la clase minoritaria (menor = mejor generación) ---")
    tabla = pd.DataFrame(resultados).T[["n_z", "jsd_min"]]
    print(tabla.to_string())
    delta = resultados["base"]["jsd_min"] - resultados["mejorado"]["jsd_min"]
    print(f"\nDelta JSD minoritaria (base - mejorado) = {delta:+.4f}")
    print("Positivo = el modelo mejorado genera mejor la clase minoritaria.")
    print(f"\nFiguras -> {dest}/exp_latente_[base|mejorado]_{ds}.png")


if __name__ == "__main__":
    main()