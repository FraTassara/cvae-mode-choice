#!/usr/bin/env python3
"""
Paso 0 — Verificar que el entorno está listo antes de correr nada.

Comprueba: dependencias Python, dependencias R, presencia de los datos crudos,
y que las columnas que menciona config.yaml existan de verdad en cada CSV.

Ese último chequeo es el que más tiempo ahorra: una columna mal escrita en
config.yaml se detecta acá y no a los 40 minutos de estimación.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import cargar  # noqa: E402

PAQUETES_R = ["apollo", "yaml", "here", "optparse", "digest"]


def check(ok: bool, mensaje: str) -> bool:
    print(f"  [{'ok' if ok else '--'}] {mensaje}")
    return ok


def main() -> int:
    cfg = cargar()
    cfg.rutas.crear()
    fallos = 0

    print("\nPython")
    for mod in ["numpy", "pandas", "sklearn", "yaml", "tensorflow", "scipy"]:
        try:
            __import__(mod)
            check(True, mod)
        except ImportError:
            fallos += not check(False, f"{mod} — falta (pip install -r requirements.txt)")

    print("\nR")
    # shutil.which funciona igual en Windows, Mac y Linux (a diferencia del
    # comando 'which', que no existe en Windows).
    rscript = shutil.which("Rscript")
    if rscript is None:
        # No es un fallo fatal: R solo hace falta para el paso 3. El resto del
        # flujo Python (pasos 1, 2 y 4) corre sin R.
        check(False, "Rscript no está en el PATH (necesario solo para el paso 3, en R)")
    else:
        check(True, "Rscript")
        for pkg in PAQUETES_R:
            r = subprocess.run(
                [rscript, "-e",
                 f"quit(status = !requireNamespace('{pkg}', quietly=TRUE))"],
                capture_output=True)
            if r.returncode != 0:
                fallos += not check(False, f"{pkg} — falta (install.packages('{pkg}'))")
            else:
                check(True, pkg)

    print("\nDatos crudos y columnas")
    for ds in cfg.datasets:
        d = cfg.dataset(ds)
        ruta = cfg.ruta_raw(ds)
        if not ruta.exists():
            fallos += not check(False, f"{ds}: falta {ruta.name}")
            continue

        df = pd.read_csv(ruta, sep=d["sep"], nrows=5)
        columnas = set(df.columns)

        necesarias = {d["choice_col"]}
        for alt in d["modelo"]["alternativas"]:
            necesarias.update(alt["atributos"].values())
            if alt.get("avail"):
                necesarias.add(alt["avail"])
        necesarias.update(d.get("cvae", {}).get("variables", {}) or {})

        faltan = sorted(necesarias - columnas)
        if faltan:
            fallos += not check(False, f"{ds}: faltan columnas {faltan}")
        else:
            check(True, f"{ds}: {len(columnas)} columnas, todas presentes")

    print()
    if fallos:
        print(f"{fallos} problema(s). Corrige antes de seguir.")
        return 1
    print("Entorno listo. Siguiente: make prueba")
    return 0


if __name__ == "__main__":
    sys.exit(main())