"""
Lector de `config.yaml` y resolución de rutas.

Regla del repo: ningún script contiene rutas, semillas ni especificaciones de
modelo. Todo sale de aquí, y este módulo es el espejo Python de `R/config.R`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

#: Raíz del repo: dos niveles arriba de este archivo (python/cvae/config.py).
RAIZ = Path(__file__).resolve().parents[2]
CONFIG_PATH = RAIZ / "config.yaml"


@dataclass(frozen=True)
class Rutas:
    raw: Path
    folds: Path
    processed: Path
    results: Path
    logs: Path

    def crear(self) -> None:
        for p in (self.raw, self.folds, self.processed, self.results, self.logs):
            p.mkdir(parents=True, exist_ok=True)


class Config:
    """Acceso tipado a `config.yaml`."""

    def __init__(self, path: Path | str = CONFIG_PATH):
        self.path = Path(path)
        with open(self.path, encoding="utf-8") as f:
            self._raw_text = f.read()
        self.d: Dict[str, Any] = yaml.safe_load(self._raw_text)

        r = self.d["proyecto"]["rutas"]
        self.rutas = Rutas(**{k: RAIZ / v for k, v in r.items()})

    # ------------------------------------------------------------------ #
    @property
    def semillas(self) -> Dict[str, int]:
        return self.d["semillas"]

    @property
    def validacion(self) -> Dict[str, Any]:
        return self.d["validacion"]

    @property
    def metodos(self) -> List[str]:
        return self.d["metodos"]

    @property
    def datasets(self) -> List[str]:
        return list(self.d["datasets"])

    def dataset(self, nombre: str) -> Dict[str, Any]:
        if nombre not in self.d["datasets"]:
            raise KeyError(
                f"Dataset '{nombre}' no está en config.yaml. "
                f"Disponibles: {self.datasets}")
        return self.d["datasets"][nombre]

    # ------------------------------------------------------------------ #
    #  Nombres de archivo: definidos UNA vez, para que Python y R no
    #  puedan desincronizarse.
    # ------------------------------------------------------------------ #
    def ruta_raw(self, ds: str) -> Path:
        return self.rutas.raw / self.dataset(ds)["archivo"]

    def ruta_original(self, ds: str) -> Path:
        return self.rutas.processed / f"{ds}_original.csv"

    def ruta_folds(self, ds: str) -> Path:
        return self.rutas.folds / f"{ds}_folds.csv"

    def ruta_train(self, ds: str, metodo: str) -> Path:
        return self.rutas.processed / f"{ds}_{metodo}_train.csv.gz"

    def ruta_estimaciones(self, ds: str) -> Path:
        return self.rutas.results / f"estimaciones_{ds}.csv"

    # ------------------------------------------------------------------ #
    def _config_de_datos(self) -> Dict[str, Any]:
        """
        Subconjunto del config que SÍ afecta a los folds y a la normalización.

        Editar la arquitectura del CVAE o la especificación del MNL no cambia
        las particiones, así que no debe invalidar los datos ya generados.
        """
        return {
            "semilla_folds": self.semillas.get("folds"),
            "validacion": self.d.get("validacion"),
            "datasets": {
                nombre: {k: v for k, v in bloque.items()
                         if k in ("archivo", "sep", "id_col", "choice_col",
                                  "clases", "filtros")}
                for nombre, bloque in self.d["datasets"].items()
            },
        }

    def hash(self) -> str:
        """Huella de la parte del config que afecta a los datos intermedios."""
        texto = json.dumps(self._config_de_datos(), sort_keys=True,
                           ensure_ascii=False)
        return hashlib.sha256(texto.encode()).hexdigest()[:12]

    def escribir_manifiesto(self, extra: Dict[str, Any] | None = None) -> Path:
        """
        Deja constancia de con qué configuración se generó `data/processed/`.
        Si el hash no coincide al correr el reporte, los datos están obsoletos.
        """
        import platform
        import sys

        m = {
            "config_hash": self.hash(),
            "python": sys.version.split()[0],
            "plataforma": platform.platform(),
            **(extra or {}),
        }
        destino = self.rutas.processed / "MANIFEST.json"
        destino.write_text(json.dumps(m, indent=2, ensure_ascii=False))
        return destino

    def verificar_manifiesto(self) -> None:
        """Aborta si los datos intermedios se generaron con otro config."""
        destino = self.rutas.processed / "MANIFEST.json"
        if not destino.exists():
            raise FileNotFoundError(
                f"No existe {destino}. Corre `make datos` primero.")
        m = json.loads(destino.read_text())
        if m["config_hash"] != self.hash():
            raise RuntimeError(
                "Cambió en config.yaml algo que afecta a los folds "
                f"({m['config_hash']} != {self.hash()}): rutas de datos, "
                "filtros, clases o parámetros de validación.\n"
                "Regenera las particiones con:\n"
                "  python python/scripts/01_folds.py\n"
                "(No hace falta borrar nada: con la misma semilla los folds "
                "salen idénticos.)")


def cargar(path: Path | str = CONFIG_PATH) -> Config:
    return Config(path)