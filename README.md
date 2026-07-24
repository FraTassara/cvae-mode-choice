# cvae-mode-choice

Aumento de datos con Conditional VAE para modelos de elección modal con clases
desbalanceadas. **Python** genera los datos sintéticos, **R + Apollo** estima los
modelos MNL.

---

## El flujo en una imagen

```
config.yaml  ────────────────────────────────────────────┐
   │  (una sola fuente de verdad: rutas, semillas,        │
   │   especificación de utilidades, tipos de variable)   │
   │                                                      │
   ├──────────────► Python ◄───────────────┐              │
   │                                       │              │
   ▼                                       │              ▼
data/raw/*.csv                             │         R lee el MISMO
   │                                       │         config.yaml
   │  [1] 01_folds.py                      │              │
   ▼                                       │              │
data/processed/{ds}_original.csv ──────────┼──────────────┤
data/folds/{ds}_folds.csv ─────────────────┼──────────────┤
   │                                       │              │
   │  [2] 02_sinteticos.py                 │              │
   ▼                                       │              │
data/processed/{ds}_{metodo}_train.csv.gz ─┼──────────────┤
                                           │              │
                                           │              ▼
                                           │      [3] R/03_estimar.R
                                           │         (Apollo)
                                           │              │
                                           │              ▼
                                           │   results/estimaciones_{ds}.csv
                                           │              │
                                           │  [4] 04_reporte.py
                                           ▼              ▼
                                  results/tabla_*.csv
```

Los tres archivos del medio son **el contrato**. Todo lo que R necesita saber de
Python está ahí y en `config.yaml`; nada más cruza la frontera.

---

## Empezar

```bash
git clone <url> && cd cvae-mode-choice
make setup        # instala dependencias Python y R
make verificar    # comprueba entorno, datos y columnas
make prueba       # corrida rápida de punta a punta (~2 min)
make all          # todo, todos los datasets
```

`make` sin argumentos muestra la ayuda.

---

## Estructura

```
config.yaml              ÚNICA fuente de verdad. Lo leen Python y R.
Makefile                 Punto de entrada. Documenta el DAG de dependencias.

data/
  raw/                   CSV originales.            [en git]
  processed/             Normalizados + sintéticos. [gitignored, regenerable]
  folds/                 Asignación de particiones. [gitignored, regenerable]

python/
  cvae/                  Paquete: CVAE, tipos de variable, muestreo latente.
  scripts/
    00_verificar.py      Chequeo de entorno y de que las columnas existen.
    01_folds.py          Paso 1.
    02_sinteticos.py     Paso 2.
    04_reporte.py        Paso 4.
    tests_contrato.py    Verifica las invariantes que R da por supuestas.

R/
  config.R               Espejo de config_repo.py. Lee el mismo YAML.
  apollo_modelo.R        Construye apollo_probabilities desde config.yaml.
  metricas.R             Métricas, con la matriz de confusión en orientación fija.
  03_estimar.R           Paso 3. Un script parametrizado, no 13 archivos.

results/                 Tablas finales.            [en git]
logs/                    Salida de Apollo, sessionInfo(). [gitignored]
```

---

## Las tres decisiones que hacen que esto sea replicable

### 1. Un solo `config.yaml`, leído por los dos lenguajes

Rutas, semillas, especificación de utilidades y tipos de variable viven en un
único archivo. Python lo lee con `yaml.safe_load`, R con `yaml::read_yaml`.

Consecuencia práctica: **cambiar el modelo es editar YAML, no código**. Añadir un
atributo a una alternativa son dos líneas en `config.yaml`; ni Python ni R se
tocan. Y no puede pasar que el filtro `AGE != 6` esté en R pero no en Python.

Si un script contiene una ruta, una semilla o un nombre de columna hard-codeado,
es un bug.

### 2. Rutas relativas y sin `setwd()`

Python resuelve con `pathlib` desde la raíz del repo; R con `here::here()`.

Ninguna ruta absoluta tipo `C:/Users/fcota/...`, y nada de
`setwd(dirname(getActiveDocumentContext()$path))`, que requiere RStudio
interactivo y hace imposible correr con `Rscript` — que es lo que necesita
`make` y lo que necesita cualquiera que clone el repo.

### 3. El sobremuestreo ocurre DENTRO de cada partición

Los folds se construyen sobre los datos originales. Para cada partición, el
método de balanceo se aplica solo a las filas de entrenamiento, y la validación
contiene únicamente filas reales.

Generar los sintéticos antes de partir en folds parece equivalente y no lo es:
una fila sintética del entrenamiento se construyó a partir de filas reales que
pueden caer en validación. `python/scripts/tests_contrato.py` verifica que esto
no pase.

---

## Qué se sube a git y qué no

| Ruta | ¿En git? | Por qué |
|---|---|---|
| `config.yaml`, `Makefile`, `python/`, `R/` | sí | es el código |
| `data/raw/` | sí, si el tamaño y la licencia lo permiten | sin esto no es replicable |
| `data/processed/`, `data/folds/` | **no** | regenerable con `make datos` |
| `results/*.csv` | sí | permite ver los resultados sin correr nada |
| `logs/` | no | ruido |
| `renv.lock`, `requirements.txt` | sí | fijan las versiones |

Si `data/raw/` es demasiado grande para git, usa
[git-lfs](https://git-lfs.com/) o publica los datos en Zenodo y deja en
`data/raw/README.md` el DOI y las instrucciones de descarga.

---

## Reproducir un resultado concreto

Cada fila de `results/estimaciones_{ds}.csv` identifica su origen con
`metodo`, `repeticion` y `fold`. Para rehacer una sola:

```bash
Rscript R/03_estimar.R --dataset sm_centro --metodo CVAE --n-repeats 1 --verbose
```

Las semillas se derivan determinísticamente de las de `config.yaml` más el
índice de partición, así que la partición `(rep=3, fold=2)` da siempre los
mismos sintéticos.

`data/processed/MANIFEST.json` guarda el hash de `config.yaml` con el que se
generaron los intermedios. Si editas el config y olvidas regenerar, los scripts
abortan con un mensaje claro en vez de mezclar datos de dos versiones.

---

## Dependencias

**Python** (`requirements.txt`): numpy, pandas, scikit-learn, tensorflow, scipy,
pyyaml, imbalanced-learn.

**R** (`renv.lock`): apollo, yaml, here, optparse, digest.

`renv` fija las versiones exactas de R. Para inicializarlo la primera vez:

```r
renv::init()
renv::snapshot()
```

Y commitea `renv.lock`.

---

## Añadir un dataset

1. Deja el CSV en `data/raw/`.
2. Agrega un bloque en `config.yaml` bajo `datasets:` — copia uno existente y
   cambia columnas, alternativas y tipos de variable.
3. `make verificar` para confirmar que las columnas existen.
4. `make DATASET=nuevo all`.

No hay que tocar código en ninguno de los dos lenguajes.

---

## Estado

- Los pasos 1, 2 y 4 (Python) están probados de punta a punta.
- El paso 3 (R/Apollo) está escrito pero **sin ejecutar**: verifícalo con
  `make prueba` antes de lanzar la corrida completa.
- Antes de confiar en los números, corre
  `python python/scripts/tests_contrato.py --dataset <ds>`.
