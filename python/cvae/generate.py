"""
Generación de datos sintéticos.

Pipeline corregido (§2.1 a y b):
  1. Ajustar una elipse de confianza por clase sobre los mu del encoder.
  2. Muestrear z ALEATORIAMENTE dentro de la elipse (rechazo por Mahalanobis).
  3. Decodificar -> MinMax inverso -> revertir log(x+1) -> recortar negativos.
  4. Generar aparte binarias / empíricas / disponibilidades condicionadas al modo.
  5. Propagar ceros estructurales de los modos no disponibles.
  6. Concatenar al dataset original y guardar.
"""

import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DatasetConfig, GenerationConfig
from .data import Dataset, encode_all, load_raw
from .latent import LatentEllipse, fit_ellipses

if TYPE_CHECKING:
    from .model import CVAE


def generate(model: "CVAE", data: Dataset, gen: GenerationConfig,
             dcfg: DatasetConfig,
             ellipses: Optional[Dict[int, LatentEllipse]] = None
             ) -> Tuple[pd.DataFrame, Dict[int, LatentEllipse]]:
    """Devuelve (bloque_sintetico, elipses_usadas)."""
    rng = np.random.default_rng(gen.seed)

    # --- 1. Elipses sobre el espacio latente de TODOS los datos reales -------
    if ellipses is None:
        X_all, y_all_cat = encode_all(data)
        mu, _ = model.encode(X_all, y_all_cat)
        ellipses = fit_ellipses(mu, data.y, confidence=gen.confidence)

    n_por_clase = gen.targets(data.class_counts())
    inv_map = {v: k for k, v in data.label_map.items()}

    bloques, etiquetas = [], []
    for etiqueta_original, n in sorted(n_por_clase.items()):
        if n <= 0:
            continue
        idx = data.label_map[etiqueta_original]

        # --- 2. Muestreo estocástico dentro de la elipse ---------------------
        z = ellipses[idx].sample(n, rng)

        entrada = np.zeros((n, model.cfg.n_z + data.n_y))
        entrada[:, : model.cfg.n_z] = z
        entrada[:, model.cfg.n_z + idx] = 1.0

        # --- 3. Decodificar y revertir transformaciones ----------------------
        decodificado = model.decoder.predict(entrada, verbose=0)
        modeladas = data.inverse(decodificado)

        # --- 4. Variables externas al CVAE ----------------------------------
        externas = data.sampler.sample(idx, n, rng)

        fila = pd.concat([modeladas.reset_index(drop=True),
                          externas.reset_index(drop=True)], axis=1)
        bloques.append(fila)
        etiquetas.extend([etiqueta_original] * n)

    if not bloques:
        raise ValueError("No se generó ninguna muestra: revisa `balance_to` / `classes`.")

    synth = pd.concat(bloques, ignore_index=True)

    # --- 5. Ceros estructurales -------------------------------------------
    synth = data.sampler.propagate_structural_zeros(synth)

    synth.insert(0, dcfg.target_col, etiquetas)

    for col, valor in gen.constant_cols.items():
        synth[col] = valor

    return synth, ellipses


def augment_and_save(model: "CVAE", data: Dataset, gen: GenerationConfig,
                     dcfg: DatasetConfig,
                     ellipses: Optional[Dict[int, LatentEllipse]] = None):
    """
    Genera, concatena al original y guarda. Devuelve
    (df_aumentado, df_sintetico, df_original, elipses).
    """
    synth, ellipses = generate(model, data, gen, dcfg, ellipses)

    base = load_raw(
        DatasetConfig(
            name=dcfg.name,
            csv_path=gen.base_csv or dcfg.csv_path,
            sep=gen.base_sep or dcfg.sep,
            target_col=dcfg.target_col,
            drop_cols=gen.base_drop_cols or dcfg.drop_cols,
            keep_cols=None if gen.base_drop_cols else dcfg.keep_cols,
            positive_only=dcfg.positive_only,
            valid_labels=dcfg.valid_labels,
        )
    )

    faltan = [c for c in base.columns if c not in synth.columns]
    if faltan:
        raise ValueError(
            f"Al bloque sintético le faltan columnas del original: {faltan}. "
            "Revisa `constant_cols` / `base_drop_cols` en la GenerationConfig."
        )
    synth = synth[base.columns]

    out = pd.concat([base, synth], axis=0, ignore_index=True)
    if gen.add_id_col:
        out[gen.add_id_col] = range(len(out))

    os.makedirs(os.path.dirname(gen.output_csv) or ".", exist_ok=True)
    out.to_csv(gen.output_csv, index=False)
    print(f"[generate] conf={gen.confidence:.0%}  {len(synth)} filas sintéticas "
          f"-> {gen.output_csv}  (total {len(out)})")

    return out, synth, base, ellipses
