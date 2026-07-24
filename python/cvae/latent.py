"""
§2.1(a) — Muestreo estocástico en el espacio latente.

El código original hacía:

    z1 = np.linspace(-0.026, -0.01, 84)
    z2 = np.linspace(-0.25, -0.05, 84)
    for i in range(84): decoder.predict([z1[i], z2[i]])

Eso recorre un SEGMENTO DE RECTA en el plano latente: las 84 muestras están
perfectamente correlacionadas (corr(z1,z2) = ±1) y equiespaciadas. No es
muestreo de la distribución latente, y explica buena parte de la degeneración
de los datos sintéticos: el CVAE nunca se usó como modelo generativo.

Reemplazo: por clase, se ajusta una normal multivariada a los mu del encoder y
se muestrea por rechazo dentro de la elipse de confianza al nivel `confidence`,
usando la distancia de Mahalanobis contra el cuantil chi-cuadrado.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import stats


@dataclass
class LatentEllipse:
    """Elipse de confianza de la distribución latente de UNA clase."""

    mean: np.ndarray            # (n_z,)
    cov: np.ndarray             # (n_z, n_z)
    confidence: float
    n_real: int                 # nº de observaciones reales de la clase

    @property
    def n_z(self) -> int:
        return len(self.mean)

    @property
    def threshold(self) -> float:
        """Cuantil chi-cuadrado: d_Mahalanobis^2 <= threshold define la elipse."""
        return float(stats.chi2.ppf(self.confidence, df=self.n_z))

    def mahalanobis_sq(self, z: np.ndarray) -> np.ndarray:
        d = z - self.mean
        return np.einsum("ij,jk,ik->i", d, np.linalg.pinv(self.cov), d)

    def sample(self, n: int, rng: np.random.Generator, max_iter: int = 1000) -> np.ndarray:
        """
        Muestreo por rechazo: se propone desde N(mean, cov) y se conservan solo
        los puntos dentro de la elipse. Para nivel 0.80 la tasa de aceptación es
        ~80%, así que el rechazo es barato.
        """
        if n <= 0:
            return np.empty((0, self.n_z))

        aceptados = []
        total = 0
        for _ in range(max_iter):
            faltan = n - total
            if faltan <= 0:
                break
            # Se propone con holgura para reducir el nº de iteraciones.
            prop = rng.multivariate_normal(self.mean, self.cov, size=int(faltan / 0.7) + 8)
            ok = prop[self.mahalanobis_sq(prop) <= self.threshold]
            if len(ok):
                aceptados.append(ok[:faltan])
                total += len(aceptados[-1])
        else:
            raise RuntimeError(
                f"Muestreo por rechazo no convergió tras {max_iter} iteraciones. "
                "Revisa que la covarianza latente no sea degenerada."
            )

        return np.vstack(aceptados)[:n]


def fit_ellipses(mu: np.ndarray, y, confidence: float = 0.80,
                 shrinkage: float = 1e-6) -> Dict[int, LatentEllipse]:
    """
    Ajusta una elipse por clase sobre los mu del encoder.

    `shrinkage` añade un ridge diminuto a la diagonal para que la covarianza sea
    invertible incluso con clases de muy pocas observaciones (el caso relevante:
    son justamente los modos minoritarios los que interesan).
    """
    y = np.asarray(y)
    out: Dict[int, LatentEllipse] = {}
    for cls in np.unique(y):
        m = mu[y == cls]
        if len(m) < 2:
            raise ValueError(
                f"La clase {cls} tiene {len(m)} observación(es); no se puede "
                "estimar una covarianza latente."
            )
        cov = np.cov(m, rowvar=False)
        cov = np.atleast_2d(cov) + shrinkage * np.eye(m.shape[1])
        out[int(cls)] = LatentEllipse(
            mean=m.mean(axis=0), cov=cov, confidence=confidence, n_real=len(m)
        )
    return out


def describe(ellipses: Dict[int, LatentEllipse],
             label_map: Optional[Dict[int, int]] = None) -> str:
    """Resumen legible de las elipses, para dejar constancia en el paper."""
    inv = {v: k for k, v in (label_map or {}).items()}
    lineas = ["clase  n_real   media_z        eigenvalores_cov      d2_umbral"]
    for cls, e in sorted(ellipses.items()):
        etiqueta = inv.get(cls, cls)
        ev = np.linalg.eigvalsh(e.cov)
        lineas.append(
            f"{etiqueta:>5}  {e.n_real:>6}   "
            f"({e.mean[0]:+.4f}, {e.mean[1]:+.4f})   "
            f"({ev[0]:.2e}, {ev[-1]:.2e})   {e.threshold:.3f}"
        )
    return "\n".join(lineas)
