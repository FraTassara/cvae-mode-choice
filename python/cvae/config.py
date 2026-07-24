"""
Configuración declarativa de los experimentos CVAE.

Todo lo que cambia entre datasets vive aquí como datos; el código de
`data.py`, `model.py` y `generate.py` es idéntico para todos.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .variables import VariableSchema


# --------------------------------------------------------------------------- #
#  Datos
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    """Cómo cargar y preparar la tabla de entrada."""

    name: str
    csv_path: str
    target_col: str                      # 'ICH', 'CHOICE', 'choice'...
    sep: str = ","

    keep_cols: Optional[Sequence[str]] = None
    drop_cols: Sequence[str] = ()

    positive_only: bool = True
    valid_labels: Optional[Sequence[int]] = None
    label_map: Optional[Dict[int, int]] = None

    #: §2.1(b) — tipo de cada variable. Las no declaradas son continuas.
    schema: VariableSchema = field(default_factory=VariableSchema)

    #: Split para validación del CVAE (no confundir con la CV del MNL).
    test_size: Optional[float] = 0.2
    random_state: int = 42

    def resolved_label_map(self, observed: Sequence[int]) -> Dict[int, int]:
        if self.label_map is not None:
            return dict(self.label_map)
        labels = list(self.valid_labels) if self.valid_labels else sorted(observed)
        return {orig: i for i, orig in enumerate(labels)}


# --------------------------------------------------------------------------- #
#  Modelo
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Arquitectura y entrenamiento del CVAE."""

    n_z: int = 2
    encoder_dims: Tuple[int, ...] = (10, 10)
    decoder_dim: int = 10
    activation: str = "relu"
    l2: float = 10e-7
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 100
    patience: int = 10
    #: §2.1(d) — épocas de warm-up del peso KL (0 desactiva el annealing).
    kl_warmup_epochs: int = 10
    checkpoint_dir: Optional[str] = None
    seed: int = 1234


# --------------------------------------------------------------------------- #
#  Generación de datos sintéticos
# --------------------------------------------------------------------------- #
@dataclass
class ClassSpec:
    """Cuántas muestras sintéticas generar para una clase.

    NOTA: ya no lleva rangos z1/z2. Antes se fijaban a mano leyéndolos del
    gráfico latente y se recorrían con np.linspace; ahora la región de muestreo
    es la elipse de confianza estimada de los datos (§2.1a).
    """

    label: int
    n: int


@dataclass
class GenerationConfig:
    """Cómo muestrear el decoder y cómo re-ensamblar el CSV aumentado."""

    #: Si es None, se calcula con `balance_to` (recomendado).
    classes: Optional[List[ClassSpec]] = None

    #: Estrategia automática de balanceo:
    #:   'majority'  -> igualar todas las clases a la mayoritaria
    #:   'fraction'  -> igualar a `balance_fraction` * n_mayoritaria
    #:   None        -> usar `classes` explícito
    balance_to: Optional[str] = "majority"
    balance_fraction: float = 1.0

    #: §2.1(a) / §2.7 — nivel de la elipse de confianza. Parametrizado para el
    #: análisis de sensibilidad 70/80/90/95 %.
    confidence: float = 0.80

    #: Semilla del muestreo (independiente de la del entrenamiento).
    seed: int = 2024

    base_csv: Optional[str] = None
    base_sep: Optional[str] = None
    base_drop_cols: Sequence[str] = ()
    constant_cols: Dict[str, int] = field(default_factory=dict)
    add_id_col: Optional[str] = None
    output_csv: str = "salida.csv"

    def targets(self, counts: Dict[int, int]) -> Dict[int, int]:
        """
        Devuelve {etiqueta_original -> nº de sintéticas} dado el conteo real.

        Reemplaza los conteos hard-coded del código original (`[1]*(0) + [2]*(84)
        + [4]*(1) + [5]*(17)`), que además tenían clases con n=0 por error.
        """
        if self.balance_to is None:
            if self.classes is None:
                raise ValueError("Define `classes` o `balance_to`.")
            return {c.label: c.n for c in self.classes}

        objetivo = max(counts.values()) * self.balance_fraction
        return {lab: max(int(round(objetivo - n)), 0) for lab, n in counts.items()}

    def with_confidence(self, confidence: float) -> "GenerationConfig":
        """Copia con otro nivel de elipse, para el análisis de sensibilidad §2.7."""
        import copy

        otra = copy.deepcopy(self)
        otra.confidence = confidence
        stem, _, ext = self.output_csv.rpartition(".")
        otra.output_csv = f"{stem}_conf{int(confidence * 100)}.{ext}"
        return otra


# --------------------------------------------------------------------------- #
@dataclass
class Experiment:
    data: DatasetConfig
    model: ModelConfig
    generation: Optional[GenerationConfig] = None
    title: str = ""
