"""
Construcción del CVAE.

Cambios respecto al código original de los notebooks:
  §2.1(d) KL annealing: peso beta que sube linealmente de 0 a 1 en las primeras
          épocas (warm-up), para evitar la KL inestable/creciente observada.
  §2.1(e) semillas de Python/NumPy/TF fijadas antes de construir el grafo.
  ELBO expuesto como métrica para la búsqueda de hiperparámetros (§2.1c).

Importante: `tf.compat.v1.disable_eager_execution()` debe llamarse ANTES de
crear cualquier capa. Lo hace `run_experiment.py` en la primera línea.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.layers import concatenate as concat
from tensorflow.keras.models import Model, model_from_json

from .config import ModelConfig
from .seeding import set_all_seeds


# ---------------------------------------------------------------------- #
#  §2.1(d) KL annealing
# ---------------------------------------------------------------------- #
class KLAnnealing(Callback):
    """
    Sube beta de `start` a `end` linealmente durante `warmup_epochs`.

    Con beta=0 el modelo arranca como un autoencoder puro (aprende a reconstruir),
    y el término KL entra gradualmente. Evita el posterior collapse y la KL
    creciente que se veía en las curvas de entrenamiento originales.
    """

    def __init__(self, beta, warmup_epochs: int = 10, start: float = 0.0, end: float = 1.0):
        super().__init__()
        self.beta = beta
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.start = start
        self.end = end

    def on_epoch_begin(self, epoch, logs=None):
        if self.warmup_epochs == 0:
            value = self.end
        else:
            frac = min(epoch / self.warmup_epochs, 1.0)
            value = self.start + frac * (self.end - self.start)
        K.set_value(self.beta, value)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        logs["beta"] = float(K.get_value(self.beta))


@dataclass
class CVAE:
    """Los tres modelos que comparten pesos: completo, encoder y decoder."""

    cvae: Model
    encoder: Model
    decoder: Model
    cfg: ModelConfig
    beta: object = None          # tf variable del peso KL

    # ------------------------------------------------------------------ #
    def fit(self, X_scaled, y_cat, validation_data=None, verbose: int = 1):
        callbacks = []

        if self.beta is not None:
            callbacks.append(
                KLAnnealing(self.beta, warmup_epochs=self.cfg.kl_warmup_epochs)
            )

        # EarlyStopping sobre la pérdida de validación si existe; si no, train.
        monitor = "val_loss" if validation_data is not None else "loss"
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=self.cfg.patience,
                restore_best_weights=True,
                # No cortar durante el warm-up: la pérdida sube por construcción.
                start_from_epoch=self.cfg.kl_warmup_epochs,
            )
        )

        if self.cfg.checkpoint_dir:
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(
                        self.cfg.checkpoint_dir,
                        "model-epoch{epoch:02d}-loss{loss:.2f}.hdf5",
                    ),
                    save_best_only=True,
                    save_weights_only=True,
                    monitor=monitor,
                )
            )

        return self.cvae.fit(
            [X_scaled, y_cat],
            X_scaled,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.epochs,
            verbose=verbose,
            validation_data=validation_data,
            shuffle=True,
            callbacks=callbacks,
        )

    # ------------------------------------------------------------------ #
    def encode(self, X_scaled, y_cat):
        """Devuelve (mu, log_sigma) del espacio latente."""
        return self.encoder.predict([X_scaled, y_cat], verbose=0)

    def elbo(self, X_scaled, y_cat) -> float:
        """
        -ELBO en el conjunto dado (menor es mejor). Se usa como criterio en la
        búsqueda de hiperparámetros (§2.1c). Se evalúa con beta=1.
        """
        if self.beta is not None:
            previo = float(K.get_value(self.beta))
            K.set_value(self.beta, 1.0)
        try:
            res = self.cvae.evaluate([X_scaled, y_cat], X_scaled, verbose=0)
        finally:
            if self.beta is not None:
                K.set_value(self.beta, previo)
        return float(res[0] if isinstance(res, (list, tuple)) else res)

    def save_architecture(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.cvae.to_json())


# ---------------------------------------------------------------------- #
def _sample_z(args):
    """Truco de reparametrización: z = mu + sigma * eps."""
    mu, l_sigma = args
    eps = K.random_normal(shape=(K.shape(mu)[0], K.shape(mu)[1]), mean=0.0, stddev=1.0)
    return mu + K.exp(l_sigma / 2) * eps


def build_cvae(n_x: int, n_y: int, cfg: ModelConfig, compile_model: bool = True) -> CVAE:
    """
    Construye encoder + decoder + CVAE completo.

    n_x : nº de features MODELADAS por el CVAE (las binarias/empíricas quedan fuera)
    n_y : nº de clases (dimensión de la etiqueta condicionante)
    """
    set_all_seeds(cfg.seed)

    # ---------------- Encoder ----------------
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

    # ---------------- Decoder ----------------
    dec_hidden = Dense(
        cfg.decoder_dim,
        activation=cfg.activation,
        kernel_regularizer=regularizers.l2(cfg.l2),
        name="decoder_hidden",
    )
    dec_out = Dense(n_x, activation="sigmoid", name="decoder_output")

    outputs = dec_out(dec_hidden(zc))

    dec_in = Input(shape=(cfg.n_z + n_y,), name="decoder_input")
    decoder = Model(dec_in, dec_out(dec_hidden(dec_in)), name="decoder")

    # ---------------- Pérdidas ----------------
    beta = K.variable(1.0 if cfg.kl_warmup_epochs == 0 else 0.0, name="kl_beta")

    def recon_loss(y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    def KL_loss(y_true, y_pred):
        return 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1.0 - l_sigma, axis=-1)

    def vae_loss(y_true, y_pred):
        # beta solo pondera el término KL durante el warm-up.
        return recon_loss(y_true, y_pred) + beta * KL_loss(y_true, y_pred)

    def elbo(y_true, y_pred):
        """-ELBO con beta=1, independiente del annealing. Métrica de selección."""
        return recon_loss(y_true, y_pred) + KL_loss(y_true, y_pred)

    cvae = Model([x_in, label], outputs, name="cvae")

    if compile_model:
        # `optimizers.legacy` existe en Keras 2 (TF < 2.16) y es el que conviene
        # en modo grafo. En Keras 3 desapareció, así que se cae al Adam normal.
        try:
            optim = tf.keras.optimizers.legacy.Adam(learning_rate=cfg.learning_rate)
        except (AttributeError, ImportError):
            optim = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        cvae.compile(
            optimizer=optim,
            loss=vae_loss,
            metrics=[KL_loss, recon_loss, elbo],
        )

    return CVAE(cvae=cvae, encoder=encoder, decoder=decoder, cfg=cfg, beta=beta)


def load_model(architecture_path: str, weights_path: str) -> Model:
    """Carga un modelo desde arquitectura .json + pesos .hdf5."""
    with open(architecture_path) as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)
    return model