"""
Puente entre `config.yaml` y el paquete `cvae`.

Traduce el bloque `datasets.<ds>.cvae` del YAML a los objetos de configuración
del paquete, para que la especificación del CVAE viva en el mismo archivo que la
del MNL y no en código Python.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .config import DatasetConfig, GenerationConfig, ModelConfig
from .variables import VariableSchema, VarKind, VarSpec


def _schema_desde_yaml(bloque: Dict[str, Any] | None,
                       excluir: list | None = None) -> VariableSchema:
    """
    Acepta dos formas por variable:

        SEXO: binary                                  # forma corta
        DISP1: {tipo: availability, depende: [...]}   # forma larga
    """
    specs: Dict[str, VarSpec] = {}

    # Columnas que NO debe modelar el CVAE (identificadores, variables fuera
    # del modelo, atributos de modos no incluidos). Se remuestrean de la
    # distribución empírica condicional al modo, así que siguen apareciendo en
    # el CSV sintético con valores plausibles, pero no ocupan capacidad del
    # espacio latente ni distorsionan la reconstrucción.
    for col in (excluir or []):
        specs[col] = VarSpec(kind=VarKind.EMPIRICAL)

    for col, valor in (bloque or {}).items():
        if isinstance(valor, str):
            specs[col] = VarSpec(kind=VarKind(valor))
        elif isinstance(valor, dict):
            specs[col] = VarSpec(
                kind=VarKind(valor["tipo"]),
                dependent_cols=valor.get("depende", ()),
                integer=valor.get("entero", False),
                non_negative=valor.get("no_negativo", True),
            )
        else:
            raise ValueError(f"Definición inválida para '{col}': {valor!r}")
    return VariableSchema(specs)


def configs_desde_yaml(cfg, ds: str, semilla: int):
    """Devuelve (DatasetConfig, ModelConfig, GenerationConfig) para un dataset."""
    d = cfg.dataset(ds)
    bloque = d.get("cvae", {}) or {}
    arq = bloque.get("arquitectura", {}) or {}
    gen = bloque.get("generacion", {}) or {}

    dcfg = DatasetConfig(
        name=ds,
        csv_path=str(cfg.ruta_original(ds)),   # ya normalizado por el paso 1
        sep=",",
        target_col=d["choice_col"],
        valid_labels=d["clases"],
        schema=_schema_desde_yaml(bloque.get("variables"),
                                  bloque.get("excluir")),
        positive_only=False,                   # el filtro ya se aplicó
        test_size=0.2,
        random_state=cfg.semillas["folds"],
    )

    # Keras exige enteros en las dimensiones. Se convierten aquí para tolerar
    # valores escritos como 5.0 en el YAML (p.ej. copiados de la salida de
    # 07_hiperparametros.py, donde pandas los devuelve como float).
    def _ent(clave, defecto):
        return int(arq.get(clave, defecto))

    mcfg = ModelConfig(
        n_z=_ent("n_z", 2),
        encoder_dims=tuple(int(x) for x in arq.get("encoder_dims", (10, 10))),
        decoder_dim=_ent("decoder_dim", 10),
        batch_size=_ent("batch_size", 128),
        epochs=_ent("epochs", 150),
        kl_warmup_epochs=_ent("kl_warmup_epochs", 10),
        learning_rate=float(arq.get("learning_rate", 1e-3)),
        checkpoint_dir=None,                   # no se guardan pesos por partición
        seed=cfg.semillas["cvae_entrena"],
    )

    gcfg = GenerationConfig(
        balance_to=gen.get("balance_to", "majority"),
        balance_fraction=gen.get("balance_fraction", 1.0),
        confidence=gen.get("confianza_elipse", 0.80),
        seed=semilla,
        output_csv="",                         # no se escribe a disco por partición
    )

    return dcfg, mcfg, gcfg


def generar_sinteticos_cvae(train: pd.DataFrame, cfg, ds: str, semilla: int,
                            confianza: float | None = None) -> pd.DataFrame:
    """
    Entrena un CVAE con las filas de entrenamiento de UNA partición y devuelve
    solo las filas sintéticas.

    Se entrena por partición, no una vez sobre todo el dataset: si el CVAE ve
    filas que después caen en validación, los sintéticos del entrenamiento
    contienen información del conjunto de validación.
    """
    import tensorflow as tf

    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()

    from tensorflow.keras import backend as K

    from .data import prepare_from_frame
    from .generate import generate
    from .model import build_cvae

    dcfg, mcfg, gcfg = configs_desde_yaml(cfg, ds, semilla)
    if confianza is not None:          # §2.7 análisis de sensibilidad
        gcfg.confidence = confianza

    # Cada partición arranca con un grafo limpio; si no, la memoria crece
    # linealmente con el número de particiones.
    K.clear_session()

    columnas = [c for c in train.columns if c != "row_id"]
    ds_obj = prepare_from_frame(train[columnas], dcfg)

    modelo = build_cvae(ds_obj.n_x, ds_obj.n_y, mcfg)
    modelo.fit(ds_obj.X_scaled, ds_obj.y_cat,
               validation_data=ds_obj.validation_data, verbose=0)

    synth, _ = generate(modelo, ds_obj, gcfg, dcfg)
    synth["row_id"] = -1

    faltan = set(train.columns) - set(synth.columns)
    if faltan:
        raise ValueError(
            f"[{ds}] al bloque sintético le faltan columnas: {sorted(faltan)}. "
            "Revisa `datasets.{ds}.cvae.variables` en config.yaml.")

    return synth[train.columns].reset_index(drop=True)