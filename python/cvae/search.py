"""
§2.1(c) — Búsqueda automática de hiperparámetros.

Original: hiperparámetros fijados a mano (batch 32/128/512, dims 10/11/15) sin
justificación, y los notebooks *_RS usaban KFold NO estratificado sobre datasets
con desbalance extremo, monitoreando 'loss' (que incluye el beta del annealing).

Corrección: random search con stratified k-fold (k=5, 15 iteraciones), criterio
= -ELBO en validación, y semilla distinta por iteración.

No depende de kerashypetune: se implementa directo para poder estratificar por
modo y evaluar el ELBO con beta=1.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K

from .config import ModelConfig
from .data import Dataset
from .model import build_cvae
from .seeding import set_all_seeds

#: Espacio de búsqueda por defecto.
#:
#: Réplica del `param_grid` de los notebooks originales (*_RS.ipynb), con una
#: sola adición: `kl_warmup_epochs`, porque el KL annealing es nuevo (§2.1d)
#: y no había un valor previo que justificar.
#:
#: `n_z` NO se busca: queda fijo en 2 por decisión de diseño del paper (el
#: espacio latente bidimensional es lo que permite las figuras y el análisis
#: de las secciones 3.3-3.4). El plan pide justificar los hiperparámetros
#: fijados "sin justificación", y este sí la tiene.
#:
#: `learning_rate` tampoco se busca: se mantiene en el 1e-3 original. Es
#: configurable desde config.yaml si se quiere cambiar a mano.
DEFAULT_SPACE: Dict[str, Any] = {
    "encoder_dim1": ("randint", 5, 15),
    "encoder_dim2": ("randint", 5, 15),
    "decoder_dim": ("randint", 5, 15),
    "activation": ("choice", ["relu"]),
    "epochs": ("choice", [100, 120]),
    "batch_size": ("choice", [16, 32, 64, 128, 256, 512]),
    "kl_warmup_epochs": ("choice", [0, 5, 10, 20]),   # §2.1(d), nuevo
}


def _draw(space: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    out = {}
    for k, spec in space.items():
        tipo = spec[0]
        if tipo == "randint":
            out[k] = int(rng.integers(spec[1], spec[2]))
        elif tipo == "choice":
            opciones = spec[1]
            out[k] = opciones[int(rng.integers(len(opciones)))]
        elif tipo == "uniform":
            out[k] = float(rng.uniform(spec[1], spec[2]))
        elif tipo == "loguniform":
            out[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        else:
            raise ValueError(f"Tipo de distribución desconocido: {tipo}")
    return out


def _to_model_cfg(params: Dict[str, Any], base: ModelConfig, seed: int) -> ModelConfig:
    return ModelConfig(
        n_z=params.get("n_z", base.n_z),
        encoder_dims=(params["encoder_dim1"], params["encoder_dim2"]),
        decoder_dim=params["decoder_dim"],
        activation=params.get("activation", base.activation),
        l2=base.l2,
        learning_rate=params.get("learning_rate", base.learning_rate),
        batch_size=params.get("batch_size", base.batch_size),
        epochs=params.get("epochs", base.epochs),
        patience=base.patience,
        kl_warmup_epochs=params.get("kl_warmup_epochs", base.kl_warmup_epochs),
        checkpoint_dir=None,
        seed=seed,
    )


@dataclass
class SearchResult:
    trials: pd.DataFrame
    best_params: Dict[str, Any]
    best_config: ModelConfig
    best_score: float


def random_search_cv(data: Dataset, base: Optional[ModelConfig] = None,
                     space: Optional[Dict[str, Any]] = None,
                     n_iter: int = 15, k: int = 5,
                     seed: int = 7, verbose: bool = True) -> SearchResult:
    """
    Random search con stratified k-fold sobre el -ELBO de validación.

    Devuelve un SearchResult con la tabla completa de trials (útil para el
    apéndice: R2 pidió justificar la elección de hiperparámetros).
    """
    base = base or ModelConfig()
    space = space or DEFAULT_SPACE
    rng = np.random.default_rng(seed)

    X_all, y_all_cat = data.X_scaled, data.y_cat
    y_flat = y_all_cat.argmax(axis=1)

    registros: List[Dict[str, Any]] = []

    for it in range(n_iter):
        params = _draw(space, rng)
        # §2.1(c): semilla distinta por iteración, para que la diversidad de la
        # búsqueda no quede anulada por una inicialización fija.
        seed_it = int(seed * 1000 + it)

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed_it)
        scores = []

        for fold, (tr, va) in enumerate(skf.split(X_all, y_flat)):
            K.clear_session()
            set_all_seeds(seed_it + fold)
            cfg = _to_model_cfg(params, base, seed_it + fold)
            modelo = build_cvae(data.n_x, data.n_y, cfg)
            modelo.fit(
                X_all[tr], y_all_cat[tr],
                validation_data=([X_all[va], y_all_cat[va]], X_all[va]),
                verbose=0,
            )
            scores.append(modelo.elbo(X_all[va], y_all_cat[va]))

        registro = dict(params)
        registro.update(
            iteracion=it,
            seed=seed_it,
            neg_elbo_media=float(np.mean(scores)),
            neg_elbo_std=float(np.std(scores)),
        )
        registros.append(registro)
        if verbose:
            print(f"[search] iter {it:>2}/{n_iter}  -ELBO = "
                  f"{registro['neg_elbo_media']:.3f} ± {registro['neg_elbo_std']:.3f}  {params}")

    trials = pd.DataFrame(registros).sort_values("neg_elbo_media").reset_index(drop=True)
    mejor = trials.iloc[0].to_dict()
    # `.iloc[0]` sobre un DataFrame de columnas mixtas convierte los enteros a
    # float. Se restauran según el tipo declarado en el espacio de búsqueda.
    best_params = {}
    for k_, spec_ in space.items():
        v = mejor[k_]
        if spec_[0] == "randint":
            v = int(v)
        elif spec_[0] == "choice" and all(isinstance(o, int) for o in spec_[1]):
            v = int(v)
        best_params[k_] = v

    return SearchResult(
        trials=trials,
        best_params=best_params,
        best_config=_to_model_cfg(best_params, base, base.seed),
        best_score=float(mejor["neg_elbo_media"]),
    )