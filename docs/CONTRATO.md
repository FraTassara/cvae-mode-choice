# El contrato Python ↔ R

Tres archivos cruzan la frontera. Fuera de ellos y de `config.yaml`, los dos
lenguajes no comparten nada.

## 1. `data/processed/{ds}_original.csv`

El dataset original ya filtrado. **R lee este archivo, no el crudo**, para que
los filtros (clases válidas, `AGE != 6`, etc.) sean idénticos en ambos lados.

| columna | tipo | descripción |
|---|---|---|
| `row_id` | int | 0..n-1, único. Es la llave de todo el contrato. |
| ... | | todas las columnas originales |

## 2. `data/folds/{ds}_folds.csv`

| columna | tipo | descripción |
|---|---|---|
| `row_id` | int | referencia a `{ds}_original.csv` |
| `repeticion` | int | 0..n_repeats-1 |
| `fold` | int | 0..n_splits-1 |
| `split` | str | `"train"` o `"valid"` |

Invariantes (verificadas por `tests_contrato.py`):
- train ∩ valid = ∅ en cada partición
- train ∪ valid = todas las filas
- cada fila cae en validación exactamente una vez por repetición

## 3. `data/processed/{ds}_{metodo}_train.csv.gz`

Los conjuntos de entrenamiento de todas las particiones, en formato largo.

| columna | tipo | descripción |
|---|---|---|
| `repeticion` | int | |
| `fold` | int | |
| `row_id` | int | `-1` en las filas sintéticas |
| `is_synthetic` | int | 0 = real, 1 = generada |
| ... | | mismas columnas que el original |

Invariantes:
- las filas con `is_synthetic == 0` de la partición (r, f) coinciden
  exactamente con las `split == "train"` de esa partición
- ninguna fila de validación aparece en su propio train
- sin `NaN` (Apollo los propaga silenciosamente a la verosimilitud)

## 4. Vuelta: `results/estimaciones_{ds}.csv`

Lo que R le devuelve a Python. Una fila por (método, repetición, fold).

| columna | descripción |
|---|---|
| `dataset`, `metodo`, `repeticion`, `fold` | identificación |
| `n_train`, `n_sinteticos` | tamaño del conjunto usado |
| `beta_*` | coeficientes estimados, **por nombre** |
| `se_*` | errores estándar, por nombre |
| `loglik_train`, `loglik_nula`, `rho2`, `convergio` | ajuste |
| `accuracy`, `f1_macro`, `loglik_holdout`, ... | métricas de validación |
| `vot` | valor del tiempo |

Los coeficientes se identifican **por nombre**, nunca por posición. Indexar
`b[1]`, `b[6]`, `b[9]` rompe en silencio si cambia el vector de parámetros.
