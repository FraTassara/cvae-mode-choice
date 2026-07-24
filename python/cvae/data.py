"""
Carga y preprocesamiento.

Cambios respecto al código original:
  §2.1(b) solo las variables CONTINUOUS/ZERO_INFLATED entran al CVAE; las
          zero-inflated pasan por log(x+1) antes del MinMaxScaler.
  El scaler se ajusta sobre el conjunto de entrenamiento, no sobre todo el dataset.
  Se guarda el ConditionalSampler ajustado sobre los datos reales, para generar
  después las binarias / empíricas / disponibilidades fuera del CVAE.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

from .config import DatasetConfig
from .variables import ConditionalSampler, VariableSchema


@dataclass
class Dataset:
    """Todo lo que el resto del pipeline necesita saber de los datos."""

    X_full: pd.DataFrame          # todas las columnas, escala original
    y: pd.Series                  # etiqueta recodificada 0..K-1
    modeled_cols: List[str]       # columnas que entran al CVAE
    external_cols: List[str]      # columnas generadas fuera del CVAE
    X_scaled: np.ndarray
    y_cat: np.ndarray
    scaler: MinMaxScaler
    schema: VariableSchema
    sampler: ConditionalSampler
    label_map: Dict[int, int]

    X_val_scaled: Optional[np.ndarray] = None
    y_val_cat: Optional[np.ndarray] = None

    @property
    def n_x(self) -> int:
        return self.X_scaled.shape[1]

    @property
    def n_y(self) -> int:
        return self.y_cat.shape[1]

    @property
    def validation_data(self):
        if self.X_val_scaled is None:
            return None
        return ([self.X_val_scaled, self.y_val_cat], self.X_val_scaled)

    def class_counts(self) -> Dict[int, int]:
        """Conteo por etiqueta ORIGINAL (1, 2, 4, 5...)."""
        inv = {v: k for k, v in self.label_map.items()}
        return {inv[k]: int(v) for k, v in self.y.value_counts().items()}

    def inverse(self, decoded: np.ndarray) -> pd.DataFrame:
        """MinMax inverso + revertir log(x+1) + recortes/redondeos."""
        df = pd.DataFrame(self.scaler.inverse_transform(decoded), columns=self.modeled_cols)
        return self.schema.backward(df)


def load_raw(cfg: DatasetConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path, sep=cfg.sep)

    if cfg.keep_cols is not None:
        df = df[list(cfg.keep_cols)]
    elif cfg.drop_cols:
        df = df.drop(columns=list(cfg.drop_cols))

    if cfg.positive_only:
        df = df[df[cfg.target_col] > 0]
    if cfg.valid_labels is not None:
        df = df[df[cfg.target_col].isin(list(cfg.valid_labels))]

    return df.reset_index(drop=True)


def prepare(cfg: DatasetConfig) -> Dataset:
    df = load_raw(cfg)

    X = df.loc[:, df.columns != cfg.target_col].astype("float32")
    y = df[cfg.target_col]

    label_map = cfg.resolved_label_map(sorted(y.unique()))
    y = y.map(label_map).astype(int)

    schema = cfg.schema
    modeled = schema.modeled_cols(X.columns)
    external = schema.external_cols(X.columns)

    if not modeled:
        raise ValueError("Ninguna columna quedó marcada como modelada por el CVAE.")

    # Los generadores de variables externas se ajustan sobre TODOS los datos
    # reales: son estimaciones de proporciones/distribuciones, no un modelo
    # entrenado, así que no hay riesgo de sobreajuste al holdout del CVAE.
    sampler = ConditionalSampler.fit(X, y, schema)

    # §2.1(b): log(x+1) en las zero-inflated ANTES de escalar.
    X_mod = schema.forward(X[modeled])

    idx_tr, idx_val = np.arange(len(X_mod)), None
    if cfg.test_size:
        idx_tr, idx_val = train_test_split(
            np.arange(len(X_mod)),
            stratify=y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
        )

    scaler = MinMaxScaler().fit(X_mod.iloc[idx_tr])

    return Dataset(
        X_full=X,
        y=y,
        modeled_cols=modeled,
        external_cols=external,
        X_scaled=scaler.transform(X_mod.iloc[idx_tr]),
        y_cat=to_categorical(y.iloc[idx_tr], num_classes=len(label_map)),
        scaler=scaler,
        schema=schema,
        sampler=sampler,
        label_map=label_map,
        X_val_scaled=None if idx_val is None else scaler.transform(X_mod.iloc[idx_val]),
        y_val_cat=None if idx_val is None else to_categorical(
            y.iloc[idx_val], num_classes=len(label_map)
        ),
    )


def encode_all(cfg_dataset: Dataset) -> tuple:
    """Matriz escalada + one-hot de TODAS las filas (para ajustar las elipses)."""
    X_mod = cfg_dataset.schema.forward(cfg_dataset.X_full[cfg_dataset.modeled_cols])
    return (
        cfg_dataset.scaler.transform(X_mod),
        to_categorical(cfg_dataset.y, num_classes=len(cfg_dataset.label_map)),
    )


def prepare_from_frame(df: pd.DataFrame, cfg: DatasetConfig) -> Dataset:
    """
    Igual que `prepare()` pero desde un DataFrame ya cargado, sin leer disco.

    Necesario para el arnés de CV (§3.2): el CVAE debe entrenarse SOLO con el
    fold de entrenamiento, no con el dataset completo.
    """
    df = df.copy()
    if cfg.positive_only:
        df = df[df[cfg.target_col] > 0]
    if cfg.valid_labels is not None:
        df = df[df[cfg.target_col].isin(list(cfg.valid_labels))]
    df = df.reset_index(drop=True)

    cols = [c for c in df.columns
            if c == cfg.target_col or c not in (cfg.drop_cols or ())]
    return _prepare_frame(df[cols], cfg)


def _prepare_frame(df: pd.DataFrame, cfg: DatasetConfig) -> Dataset:
    """Núcleo compartido por `prepare` y `prepare_from_frame`."""
    X = df.loc[:, df.columns != cfg.target_col].astype("float32")
    y = df[cfg.target_col]

    label_map = cfg.resolved_label_map(sorted(y.unique()))
    y = y.map(label_map).astype(int)

    schema = cfg.schema
    modeled = schema.modeled_cols(X.columns)
    external = schema.external_cols(X.columns)
    if not modeled:
        raise ValueError("Ninguna columna quedó marcada como modelada por el CVAE.")

    sampler = ConditionalSampler.fit(X, y, schema)
    X_mod = schema.forward(X[modeled])

    idx_tr, idx_val = np.arange(len(X_mod)), None
    if cfg.test_size:
        idx_tr, idx_val = train_test_split(
            np.arange(len(X_mod)), stratify=y,
            test_size=cfg.test_size, random_state=cfg.random_state)

    scaler = MinMaxScaler().fit(X_mod.iloc[idx_tr])

    return Dataset(
        X_full=X, y=y, modeled_cols=modeled, external_cols=external,
        X_scaled=scaler.transform(X_mod.iloc[idx_tr]),
        y_cat=to_categorical(y.iloc[idx_tr], num_classes=len(label_map)),
        scaler=scaler, schema=schema, sampler=sampler, label_map=label_map,
        X_val_scaled=None if idx_val is None else scaler.transform(X_mod.iloc[idx_val]),
        y_val_cat=None if idx_val is None else to_categorical(
            y.iloc[idx_val], num_classes=len(label_map)),
    )
