"""
CVAE con KL annealing POR CLASE (class-aware KL weighting, CAW).

Módulo experimental acoplable. NO modifica el paquete `cvae`: importa lo que
puede y solo reimplementa la parte de la pérdida que cambia.

Idea (Wang et al. 2025, CAMIL): para las clases minoritarias el término KL
tiende a dominar y empuja su representación latente hacia el prior N(0, I),
colapsando su diversidad ("posterior collapse" de la minoría). La solución es
RELAJAR el KL de las clases raras —darles un peso KL más bajo— para que su
latente conserve estructura y el decoder genere muestras más diversas.

El peso por clase se define como:

    w_c = kl_min_scale + (1 - kl_min_scale) * (n_c / n_max)

donde n_c es el nº de observaciones de la clase c y n_max el de la mayoritaria.
Así la clase mayoritaria tiene w = 1 (KL completo) y la minoritaria tiene
w = kl_min_scale (KL relajado). Con kl_min_scale = 1 se recupera el CVAE
estándar (todas las clases con KL completo), lo que permite comparar.

Se combina con el annealing temporal del modelo base: el beta global sigue
subiendo de 0 a 1 en el warm-up; el peso por clase lo multiplica.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.layers import concatenate as concat
from tensorflow.keras.models import Model

# Reutilizamos TODO lo que no cambia del paquete base.
from cvae.config import ModelConfig
from cvae.model import CVAE, _sample_z
from cvae.seeding import set_all_seeds


def pesos_kl_por_clase(y_int, kl_min_scale: float) -> dict:
    """
    Peso KL de cada clase, interpolando entre kl_min_scale (minoritaria) y 1
    (mayoritaria) según su frecuencia. Devuelve {clase_interna: peso}.
    """
    y_int = np.asarray(y_int)
    clases, conteos = np.unique(y_int, return_counts=True)
    n_max = conteos.max()
    return {int(c): kl_min_scale + (1.0 - kl_min_scale) * (n / n_max)
            for c, n in zip(clases, conteos)}


def build_cvae_class_aware(n_x: int, n_y: int, cfg: ModelConfig,
                           peso_por_clase: dict,
                           compile_model: bool = True) -> CVAE:
    """
    Igual que cvae.model.build_cvae, pero el término KL se pondera por clase.

    peso_por_clase : {clase_interna: peso_kl} de pesos_kl_por_clase().
                     La clase se lee del one-hot y_true, así que NO hace falta
                     cambiar la firma de la pérdida ni el flujo de datos.
    """
    set_all_seeds(cfg.seed)

    # Vector de pesos indexado por clase interna (orden 0..n_y-1).
    w_vec = np.ones(n_y, dtype="float32")
    for c, w in peso_por_clase.items():
        if 0 <= c < n_y:
            w_vec[c] = w
    w_tf = K.constant(w_vec, name="kl_class_weights")

    # ---------------- Encoder (idéntico al base) ----------------
    x_in = Input(shape=(n_x,), name="encoder_input_x")
    label = Input(shape=(n_y,), name="encoder_input_y")
    h = concat([x_in, label])
    for i, dim in enumerate(cfg.encoder_dims, start=1):
        h = Dense(dim, activation=cfg.activation, name=f"encoder_hidden_{i}")(h)
    mu = Dense(cfg.n_z, activation="linear", name="latent_mu")(h)
    l_sigma = Dense(cfg.n_z, activation="linear", name="latent_sigma")(h)
    z = Lambda(_sample_z, output_shape=(cfg.n_z,), name="z")([mu, l_sigma])
    zc = concat([z, label])
    encoder = Model([x_in, label], [mu, l_sigma], name="encoder")

    # ---------------- Decoder (idéntico al base) ----------------
    dec_hidden = Dense(cfg.decoder_dim, activation=cfg.activation,
                       kernel_regularizer=regularizers.l2(cfg.l2),
                       name="decoder_hidden")
    dec_out = Dense(n_x, activation="sigmoid", name="decoder_output")
    outputs = dec_out(dec_hidden(zc))
    dec_in = Input(shape=(cfg.n_z + n_y,), name="decoder_input")
    decoder = Model(dec_in, dec_out(dec_hidden(dec_in)), name="decoder")

    # ---------------- Pérdidas (aquí está el cambio) ----------------
    beta = K.variable(1.0 if cfg.kl_warmup_epochs == 0 else 0.0, name="kl_beta")

    def recon_loss(y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    def KL_loss(y_true, y_pred):
        return 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1.0 - l_sigma, axis=-1)

    def _peso_kl_de_la_muestra(y_true):
        # y_true es la etiqueta one-hot (misma que se pasa como 'x' a reconstruir
        # NO: aquí y_true es la SALIDA reconstruida). El one-hot de clase viaja
        # por el input `label`, que está disponible como tensor del grafo.
        # Producto punto one-hot . w_vec -> peso de cada fila del batch.
        return K.sum(label * w_tf, axis=-1)

    def vae_loss(y_true, y_pred):
        w = _peso_kl_de_la_muestra(y_true)          # peso por observación
        return recon_loss(y_true, y_pred) + beta * w * KL_loss(y_true, y_pred)

    def elbo(y_true, y_pred):
        # -ELBO estándar (sin pesos de clase, beta=1): métrica comparable con
        # el modelo base para la selección de hiperparámetros.
        return recon_loss(y_true, y_pred) + KL_loss(y_true, y_pred)

    cvae = Model([x_in, label], outputs, name="cvae_class_aware")

    if compile_model:
        try:
            optim = tf.keras.optimizers.legacy.Adam(learning_rate=cfg.learning_rate)
        except (AttributeError, ImportError):
            optim = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        cvae.compile(optimizer=optim, loss=vae_loss,
                     metrics=[KL_loss, recon_loss, elbo])

    return CVAE(cvae=cvae, encoder=encoder, decoder=decoder, cfg=cfg, beta=beta)
