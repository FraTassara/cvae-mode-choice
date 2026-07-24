"""
§2.1(b) — Tratamiento diferenciado de variables por tipo.

El código original aplicaba MinMaxScaler a TODAS las columnas por igual: a tiempos
de viaje zero-inflated, a binarias como SEXO y a variables de disponibilidad. Eso
produce muestras sintéticas donde una binaria vale 0.63 y un tiempo de viaje de un
modo no disponible es positivo.

Aquí las variables se clasifican y cada tipo recibe su tratamiento:

  CONTINUOUS     modelada por el CVAE, MinMax simple.
  ZERO_INFLATED  modelada por el CVAE, log(x+1) -> MinMax; al revertir exp(x)-1.
  BINARY         NO modelada. Se genera con Bernoulli condicional al modo.
  EMPIRICAL      NO modelada. Se remuestrea de la distribución empírica por modo.
  AVAILABILITY   NO modelada. Bernoulli por modo + propagación de ceros estructurales.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class VarKind(str, Enum):
    CONTINUOUS = "continuous"
    ZERO_INFLATED = "zero_inflated"
    BINARY = "binary"
    EMPIRICAL = "empirical"
    AVAILABILITY = "availability"


#: Tipos que entran como input/output del CVAE.
MODELED = {VarKind.CONTINUOUS, VarKind.ZERO_INFLATED}


@dataclass
class VarSpec:
    kind: VarKind = VarKind.CONTINUOUS
    #: Solo para AVAILABILITY: columnas que se ponen a 0 si esta vale 0.
    dependent_cols: Sequence[str] = ()
    #: Solo para CONTINUOUS/ZERO_INFLATED: redondear a entero al generar.
    integer: bool = False
    #: Recortar a >= 0 al revertir (tiempos y costos no pueden ser negativos).
    non_negative: bool = True


@dataclass
class VariableSchema:
    """Mapa columna -> VarSpec. Las columnas no declaradas son CONTINUOUS."""

    specs: Dict[str, VarSpec] = field(default_factory=dict)

    def kind(self, col: str) -> VarKind:
        return self.specs.get(col, VarSpec()).kind

    def spec(self, col: str) -> VarSpec:
        return self.specs.get(col, VarSpec())

    def modeled_cols(self, columns: Sequence[str]) -> List[str]:
        """Columnas que entran al CVAE."""
        return [c for c in columns if self.kind(c) in MODELED]

    def external_cols(self, columns: Sequence[str]) -> List[str]:
        """Columnas generadas fuera del CVAE."""
        return [c for c in columns if self.kind(c) not in MODELED]

    def availability_cols(self, columns: Sequence[str]) -> List[str]:
        return [c for c in columns if self.kind(c) == VarKind.AVAILABILITY]

    # ------------------------------------------------------------------ #
    #  Transformaciones previas al escalado
    # ------------------------------------------------------------------ #
    def forward(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica log(x+1) a las zero-inflated. Se llama ANTES del MinMaxScaler."""
        out = X.copy()
        for col in out.columns:
            if self.kind(col) == VarKind.ZERO_INFLATED:
                out[col] = np.log1p(np.clip(out[col].values, 0, None))
        return out

    def backward(self, X: pd.DataFrame) -> pd.DataFrame:
        """Revierte log(x+1) y aplica recortes/redondeos. DESPUÉS del inverse_transform."""
        out = X.copy()
        for col in out.columns:
            spec = self.spec(col)
            if spec.kind == VarKind.ZERO_INFLATED:
                out[col] = np.expm1(out[col].values)
            if spec.non_negative and spec.kind in MODELED:
                out[col] = np.clip(out[col].values, 0, None)
            if spec.integer:
                out[col] = out[col].round()
        return out


# ---------------------------------------------------------------------- #
#  Generadores condicionales para las variables externas al CVAE
# ---------------------------------------------------------------------- #
@dataclass
class ConditionalSampler:
    """
    Estima y muestrea las variables que NO pasan por el CVAE, condicionando
    al modo elegido. Se ajusta sobre los datos REALES.
    """

    schema: VariableSchema
    #: col -> {clase -> P(x=1)}   para BINARY y AVAILABILITY
    probs: Dict[str, Dict[int, float]] = field(default_factory=dict)
    #: col -> {clase -> array de valores observados}   para EMPIRICAL
    pools: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def fit(cls, X: pd.DataFrame, y: pd.Series, schema: VariableSchema) -> "ConditionalSampler":
        probs: Dict[str, Dict[int, float]] = {}
        pools: Dict[str, Dict[int, np.ndarray]] = {}

        for col in schema.external_cols(X.columns):
            kind = schema.kind(col)
            if kind in (VarKind.BINARY, VarKind.AVAILABILITY):
                probs[col] = {
                    int(c): float((X.loc[y == c, col] > 0.5).mean())
                    for c in sorted(y.unique())
                }
            elif kind == VarKind.EMPIRICAL:
                pools[col] = {
                    int(c): X.loc[y == c, col].to_numpy() for c in sorted(y.unique())
                }
        return cls(schema=schema, probs=probs, pools=pools)

    def sample(self, class_index: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Genera `n` filas de las variables externas para una clase."""
        out = {}
        for col, p_by_class in self.probs.items():
            p = p_by_class.get(class_index, 0.0)
            out[col] = rng.binomial(1, p, size=n).astype(float)
        for col, pool_by_class in self.pools.items():
            pool = pool_by_class.get(class_index)
            if pool is None or len(pool) == 0:
                out[col] = np.zeros(n)
            else:
                out[col] = rng.choice(pool, size=n, replace=True)
        return pd.DataFrame(out, index=range(n))

    def propagate_structural_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Si la disponibilidad de un modo es 0, sus atributos (tiempo, costo,
        espera de ese modo) se fuerzan a 0. Sin esto el CSV sintético contiene
        tiempos de viaje positivos para modos declarados no disponibles.
        """
        out = df.copy()
        for col in self.schema.availability_cols(out.columns):
            deps = [c for c in self.schema.spec(col).dependent_cols if c in out.columns]
            if not deps:
                continue
            mask = out[col].to_numpy() < 0.5
            out.loc[mask, deps] = 0.0
        return out
