#!/usr/bin/env python3
"""
Paso 7 — Búsqueda de hiperparámetros del CVAE (§2.1c del plan de revisores).

Random search con validación cruzada ESTRATIFICADA sobre el -ELBO de
validación, con semilla distinta por iteración. Reemplaza los hiperparámetros
fijados a mano.

El script NO modifica config.yaml: imprime la mejor configuración y guarda la
tabla completa de intentos. Tú decides si la copias al config y reejecutas.

ADVERTENCIA DE TIEMPO: entrena n_iter x k modelos (por defecto 15 x 5 = 75).
En CPU puede tardar de 20 minutos a varias horas según el dataset. Prueba
primero con  --n-iter 3 --k 2  para estimar cuánto tarda uno.

Uso:
    python python/scripts/07_hiperparametros.py --dataset sm_centro
    python python/scripts/07_hiperparametros.py --dataset sm_centro --n-iter 3 --k 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvae.config_repo import cargar  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-iter", type=int, default=15,
                    help="configuraciones a probar (plan: 15)")
    ap.add_argument("--k", type=int, default=5,
                    help="pliegues del k-fold estratificado (plan: 5)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    from cvae.data import prepare_from_frame
    from cvae.pipeline import configs_desde_yaml
    from cvae.search import DEFAULT_SPACE, random_search_cv

    cfg = cargar()
    ds = args.dataset
    origen = cfg.ruta_original(ds)
    if not origen.exists():
        raise SystemExit(
            f"Falta {origen}.\n"
            f"Corre primero: python python/scripts/01_folds.py --dataset {ds}")

    datos = pd.read_csv(origen)
    dcfg, mcfg, _ = configs_desde_yaml(cfg, ds, cfg.semillas["cvae_muestreo"])
    columnas = [c for c in datos.columns if c != "row_id"]
    dataset = prepare_from_frame(datos[columnas], dcfg)

    print(f"\n[{ds}] Búsqueda de hiperparámetros")
    print(f"  observaciones: {len(datos)}   variables al CVAE: {dataset.n_x}")
    print(f"  {args.n_iter} configuraciones x {args.k} pliegues "
          f"= {args.n_iter * args.k} entrenamientos")
    print(f"\n  espacio de búsqueda:")
    for k_, v in DEFAULT_SPACE.items():
        print(f"    {k_:<18} {v}")
    print()

    t0 = time.time()
    res = random_search_cv(dataset, base=mcfg, n_iter=args.n_iter, k=args.k,
                           seed=args.seed, verbose=True)
    minutos = (time.time() - t0) / 60

    destino = cfg.rutas.results / f"hiperparametros_{ds}.csv"
    res.trials.to_csv(destino, index=False)

    print(f"\n{'='*62}")
    print(f"Mejores 5 configuraciones (menor -ELBO es mejor):")
    cols = [c for c in res.trials.columns if c not in ("iteracion", "seed")]
    print(res.trials.head(5)[cols].round(4).to_string(index=False))

    print(f"\nTiempo: {minutos:.1f} min   Tabla completa -> {destino.name}")

    b = res.best_params
    print(f"\n{'='*62}")
    print("VALORES ENCONTRADOS POR LA BÚSQUEDA")
    print("Copia estas líneas al bloque `cvae.arquitectura` de config.yaml, "
          f"dataset '{ds}':\n")
    print("      arquitectura:")
    print(f"        encoder_dims: [{b['encoder_dim1']}, {b['encoder_dim2']}]")
    print(f"        decoder_dim: {b['decoder_dim']}")
    print(f"        batch_size: {b['batch_size']}")
    print(f"        epochs: {b['epochs']}")
    if "kl_warmup_epochs" in b:
        print(f"        kl_warmup_epochs: {b['kl_warmup_epochs']}")

    # Los que NO se buscan se muestran aparte, con su valor actual, para que
    # quede claro que son decisiones tuyas y no resultados del algoritmo.
    no_buscados = {
        "n_z": mcfg.n_z,
        "learning_rate": mcfg.learning_rate,
    }
    no_buscados = {k: v for k, v in no_buscados.items() if k not in b}
    if no_buscados:
        print(f"\n{'-'*62}")
        print("NO SE BUSCARON — valores actuales de tu config.yaml.")
        print("Déjalos como están salvo que quieras cambiarlos a propósito:\n")
        for k, v in no_buscados.items():
            print(f"        {k}: {v}")
        if mcfg.n_z != 2:
            print(f"\n  ATENCIÓN: n_z = {mcfg.n_z}, no 2. El paper asume un "
                  "espacio latente\n  bidimensional (figuras y secciones 3.3-3.4). "
                  "Revisa si es intencional.")

    print(f"\nDespués de editar config.yaml, reejecuta desde el paso 2:")
    print(f"  python python/scripts/02_sinteticos.py --dataset {ds} --metodo CVAE")
    print(f"  Rscript R/03_estimar.R --dataset {ds} --metodo CVAE")
    print(f"  python python/scripts/04_reporte.py --dataset {ds}")


if __name__ == "__main__":
    main()