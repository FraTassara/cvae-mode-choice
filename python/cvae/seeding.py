"""
§2.1(e) — Reproducibilidad completa.

El código original hacía solo `random.seed(1234)`, que no afecta ni a NumPy ni a
TensorFlow: los pesos iniciales, el shuffle del fit y el ruido del muestreo latente
quedaban sin controlar. En la práctica NINGUNO de los resultados era reproducible.
"""

import os
import random

import numpy as np


def set_all_seeds(seed: int, deterministic_ops: bool = False) -> np.random.Generator:
    """
    Fija las semillas de Python, NumPy y TensorFlow, y devuelve un `Generator`
    de NumPy para el muestreo explícito (elipses, Bernoulli, empíricas).

    deterministic_ops=True fuerza kernels deterministas en TF. Es más lento pero
    necesario si se quiere bit-exactitud entre corridas.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)
    if deterministic_ops:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass

    return np.random.default_rng(seed)
