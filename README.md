# cvae-mode-choice

Aumento de datos con **Conditional VAE** para modelos de elección modal (MNL) con
clases desbalanceadas. **Python** genera los datos sintéticos y calcula los
diagnósticos; **R + Apollo** estima los modelos. Los dos lados se comunican por
archivos con un contrato explícito, y todo se controla desde un único
`config.yaml`.

El objetivo del proyecto es comparar métodos de balanceo (ROS, SMOTE, CVAE,
ponderación por clase) frente a una línea base sin balancear, midiendo tanto el
desempeño predictivo como la **coherencia económica de los coeficientes** — es
decir, si los signos y magnitudes de los β siguen teniendo sentido después de
aumentar los datos.

---

## El flujo en una imagen

```
config.yaml  ─────────────────────────────────────────────┐
   │  (fuente única de verdad: rutas, semillas, filtros,   │
   │   especificación de utilidades del MNL, tipos de      │
   │   variable y exclusiones del CVAE)                    │
   │                                                       │
   ▼                                                       ▼
data/raw/*.csv                                     R lee el MISMO
   │                                               config.yaml
   │  [1] 01_folds.py       normaliza + particiona         │
   ▼                                                       │
data/processed/{ds}_original.csv  ──────────────────────►  │
data/folds/{ds}_folds.csv  ─────────────────────────────►  │
   │                                                       │
   │  [2] 02_sinteticos.py  balanceo DENTRO de cada fold   │
   ▼                                                       │
data/processed/{ds}_{metodo}_train.csv.gz  ─────────────►  │
   │                                                       ▼
   │                                           [3] R/03_estimar.R
   │  (verificación del contrato antes de estimar)      Apollo MNL
   │                                                       │
   │                                                       ▼
   │                                    results/estimaciones_{ds}.csv
   │                                                       │
   │  [4] 04_reporte.py   →   [5] 05_excel.py              │
   ▼                                                       ▼
results/tabla_*.csv               results/tablas_{ds}.xlsx

Diagnósticos (independientes, al final):
   [6] 06_diagnosticos.py     JSD, gaps de atributos, ratios   (Python)
   [7] 07_hiperparametros.py  búsqueda de arquitectura del CVAE (Python)
   [8] R/08_efectos_marginales.R  elasticidades por método     (R)
```

---

## Empezar

### Requisitos

- **Python 3.11** (no 3.12+: TensorFlow < 2.16 aún no tiene ruedas para ellas).
- **R 4.x** con Apollo, para el paso de estimación.

### Instalación

```bash
git clone <url> cvae-mode-choice
cd cvae-mode-choice

# Python
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt

# R (una vez)
Rscript -e "install.packages(c('apollo','yaml','here','optparse','digest'))"
```

### Datos

Copia los CSV originales en `data/raw/` con los nombres que declara
`config.yaml`. Luego verifica que todo esté en su sitio:

```bash
python python/scripts/00_verificar.py
```

Comprueba dependencias, presencia de los CSV y —lo más útil— que **todas las
columnas mencionadas en `config.yaml` existan de verdad** en cada archivo. Una
columna mal escrita se detecta aquí y no a los cuarenta minutos de estimación.

### Correr todo

En Windows hay un script que encadena los pasos para un dataset:

```bat
.\correr_todo.bat sm_centro --con-busqueda    REM incluye búsqueda de hiperparámetros
.\correr_todo.bat sm_centro                    REM el pipeline completo
```

En Mac/Linux, o si prefieres control paso a paso, ver "Pipeline detallado".

---

## Estructura

```
config.yaml              FUENTE ÚNICA DE VERDAD. La leen Python y R.
correr_todo.bat          Encadena el pipeline para un dataset (Windows).
requirements.txt         Dependencias Python.

data/
  raw/                   CSV originales.            [en git si el tamaño lo permite]
  processed/             Normalizados + sintéticos. [gitignored, regenerable]
  folds/                 Asignación de particiones. [gitignored, regenerable]

python/
  cvae/                  Paquete: CVAE, tipos de variable, muestreo latente.
    config.py            Dataclasses de configuración.
    config_repo.py       Lector de config.yaml + manifiesto de reproducibilidad.
    variables.py         Tipos de variable y generadores condicionales.
    data.py              Carga, transformaciones, escalado.
    model.py             Encoder, decoder, pérdidas, KL annealing.
    latent.py            Elipses de confianza, muestreo por rechazo.
    generate.py          Muestreo, decodificación, re-ensamblado.
    diagnostics.py       JSD, benchmark interno, ratios, gaps de atributos.
    search.py            Random search estratificado sobre el ELBO.
    seeding.py           Semillas Python + NumPy + TensorFlow.
    plots.py             Espacio latente, distribuciones.
    pipeline.py          Puente entre config.yaml y el paquete.
  scripts/
    00_verificar.py      Chequeo de entorno y columnas.
    01_folds.py          Paso 1: normalizar + particionar.
    02_sinteticos.py     Paso 2: generar por partición.
    04_reporte.py        Paso 4: tablas finales.
    05_excel.py          Paso 5: exportar a Excel con formato.
    06_diagnosticos.py   Diagnósticos de calidad de los sintéticos.
    07_hiperparametros.py  Búsqueda de arquitectura del CVAE.
    tests_contrato.py    Verifica las invariantes que R da por supuestas.

R/
  config.R               Espejo de config_repo.py. Lee el mismo YAML.
  apollo_modelo.R        Construye apollo_probabilities desde config.yaml.
  metricas.R             Métricas, matriz de confusión en orientación fija.
  03_estimar.R           Paso 3: estimación (un script parametrizado).
  08_efectos_marginales.R  Elasticidades por método.

docs/CONTRATO.md         Especificación del contrato entre lenguajes.
results/                 Tablas y diagnósticos finales.
logs/                    Salida de Apollo, sessionInfo(). [gitignored]
```

---

## Los métodos de balanceo

Cada método es una forma distinta de corregir el desbalance de clases. Se
declaran en `config.yaml` bajo `metodos:`.

| método | qué hace | genera datos |
|---|---|---|
| `original` | nada; línea base para comparar | no |
| `ROS` | duplica filas reales de las clases minoritarias | sí (copias) |
| `SMOTE` | interpola entre vecinos cercanos de la clase minoritaria | sí (interpolación) |
| `CVAE` | aprende la distribución y muestrea observaciones nuevas | sí (generativo) |
| `class_weights` | pondera la verosimilitud por el inverso de la frecuencia | no (pesos) |

El punto de comparación no es solo cuál predice mejor, sino cuál lo hace **sin
destruir la interpretación económica** de los coeficientes.

---

## Las decisiones de diseño que hacen esto replicable

### 1. Un solo `config.yaml`, leído por los dos lenguajes

Rutas, semillas, filtros, especificación de utilidades del MNL, tipos de
variable y exclusiones del CVAE viven en un único archivo. Python lo lee con
`yaml.safe_load`, R con `yaml::read_yaml`.

Consecuencia práctica: **cambiar el modelo es editar YAML, no código**.
`R/apollo_modelo.R` construye `apollo_probabilities` por metaprogramación desde
ese archivo, así que añadir un atributo a una alternativa son dos líneas en el
config y ni Python ni R se tocan. Y un filtro (por ejemplo `AGE != 6` en
Swissmetro) no puede quedar aplicado en un lado y no en el otro.

Regla del repo: si un script contiene una ruta, una semilla o un nombre de
columna escrito a mano, es un bug.

### 2. Rutas relativas, sin `setwd()`

Python resuelve con `pathlib` desde la raíz del repo; R con `here::here()`.
Ninguna ruta absoluta, y nada de `setwd(getActiveDocumentContext()$path)`, que
requiere RStudio interactivo e impide correr con `Rscript` — que es lo que
necesita cualquiera que clone el repo.

### 3. El sobremuestreo ocurre DENTRO de cada partición

Los folds se construyen sobre los datos **originales**. Para cada partición, el
balanceo se aplica solo a las filas de entrenamiento, y la validación contiene
únicamente filas reales.

Generar los sintéticos antes de partir en folds parece equivalente y no lo es:
una fila sintética del entrenamiento se genera a partir de filas reales que
pueden caer en validación, y el modelo terminaría evaluándose sobre datos que
influyeron en su propio entrenamiento. Por eso, en el caso del CVAE, se entrena
un modelo nuevo por partición. `tests_contrato.py` verifica que ninguna fila de
validación aparezca en su propio train.

### 4. El CVAE modela solo las variables que corresponden

`config.yaml` declara, por dataset, una lista `excluir` con las columnas que el
CVAE **no** debe modelar: identificadores, variables fuera del modelo y
atributos de modos que no forman parte de la especificación. Esas columnas se
remuestrean de la distribución empírica condicional al modo, así que siguen
apareciendo en el CSV sintético con valores plausibles, pero no ocupan capacidad
del espacio latente ni distorsionan la reconstrucción.

### 5. Un manifiesto que solo salta cuando hace falta

`data/processed/MANIFEST.json` guarda un hash de la parte de `config.yaml` que
afecta a los datos intermedios (rutas, filtros, clases, semilla y parámetros de
validación). Editar la arquitectura del CVAE o la especificación del MNL **no**
invalida los folds ya generados; cambiar las clases o la semilla, sí. Si hay
desajuste, los scripts abortan con un mensaje claro en vez de mezclar datos de
dos versiones.

---

## Correcciones metodológicas al CVAE

El generador incorpora varias correcciones respecto de la implementación inicial
(detalladas en el documento de resumen). Las principales:

- **Muestreo estocástico en elipses de confianza** en lugar de recorrer una
  recta con `np.linspace`. El código anterior generaba puntos sobre un segmento
  del plano latente (correlación ±1 exacta); ahora se ajusta una normal por clase
  y se muestrea por rechazo dentro de la elipse al nivel de confianza indicado.
- **Tratamiento por tipo de variable**: continuas y zero-inflated (con
  `log(x+1)`) pasan por el CVAE; binarias, empíricas y disponibilidades se
  generan aparte condicionando al modo, con propagación de ceros estructurales.
- **KL annealing**: el peso del término KL sube de 0 a 1 durante las primeras
  épocas, para estabilizar el entrenamiento.
- **Semillas completas** (Python, NumPy, TensorFlow) para reproducibilidad.
- **Validación cruzada real**: folds estratificados sobre datos originales,
  balanceo solo en el train, evaluación sobre filas reales.

---

## Pipeline detallado (paso a paso)

Reemplaza `sm_centro` por el dataset que quieras (`lc_centro`, `whalen`,
`swissmetro`).

```bash
# 0. Verificar entorno y columnas
python python/scripts/00_verificar.py

# 1. Normalizar y construir folds
python python/scripts/01_folds.py --dataset sm_centro

# (opcional) Buscar la arquitectura del CVAE. NO escribe en config.yaml:
# imprime el bloque para copiar y guarda results/hiperparametros_sm_centro.csv.
python python/scripts/07_hiperparametros.py --dataset sm_centro

# 2. Generar conjuntos de entrenamiento (todos los métodos del config)
python python/scripts/02_sinteticos.py --dataset sm_centro

# 2b. Verificar el contrato ANTES de estimar
python python/scripts/tests_contrato.py --dataset sm_centro

# 3. Estimar en R con Apollo (acumula: no borra métodos ya estimados)
Rscript R/03_estimar.R --dataset sm_centro

# 4-5. Tablas y Excel
python python/scripts/04_reporte.py --dataset sm_centro
python python/scripts/05_excel.py --dataset sm_centro

# 6, 8. Diagnósticos
python python/scripts/06_diagnosticos.py --dataset sm_centro
Rscript R/08_efectos_marginales.R --dataset sm_centro
```

Para estimar un solo método (por ejemplo tras cambiar la arquitectura del CVAE),
`03_estimar.R` acepta `--metodo CVAE` y **reemplaza solo esas filas**,
conservando los demás métodos ya estimados. Después hay que rehacer los pasos 4
y 5, que consolidan todos los métodos.

### Análisis de sensibilidad de elipses

El nivel de confianza de la elipse se puede sobrescribir por línea de comandos;
cada nivel se guarda como un método aparte:

```bash
python python/scripts/02_sinteticos.py --dataset sm_centro --metodo CVAE --confianza 0.70
Rscript R/03_estimar.R --dataset sm_centro --metodo CVAE_conf70
```

---

## Qué produce

En `results/`, por dataset:

- `tablas_{ds}.xlsx` — tres pestañas: **Predictiva** (accuracy, F1, log-
  verosimilitud, VOT por método), **Coeficientes** (media, S.D. entre
  particiones, S.E. del estimador) y **Cambios de signo** (coeficientes que
  invierten el signo esperado, con resaltado).
- `estimaciones_{ds}.csv` — una fila por (método, repetición, fold), materia
  prima de las tablas.
- `jsd_{ds}.csv`, `sobremuestreo_{ds}.csv`, `gaps_atributos_{ds}.csv` — los
  diagnósticos. El de **gaps** es el que explica el mecanismo: si la diferencia
  de atributos entre alternativa elegida y no elegidas se distorsiona en los
  datos sintéticos, ahí está el origen de las reversiones de signo del MNL.
- `hiperparametros_{ds}.csv` — la tabla de intentos de la búsqueda, si se corrió.
- `efectos_marginales_{ds}.csv` — elasticidades por método.

---

## El contrato Python ↔ R

Cuatro archivos cruzan la frontera entre lenguajes; fuera de ellos y de
`config.yaml`, los dos lados no comparten nada. Está documentado en
`docs/CONTRATO.md` y verificado por `tests_contrato.py`. En resumen:

| archivo | rol |
|---|---|
| `{ds}_original.csv` | dataset ya filtrado, con `row_id`. R lee este, no el crudo. |
| `{ds}_folds.csv` | asignación de particiones (`row_id`, `repeticion`, `fold`, `split`). |
| `{ds}_{metodo}_train.csv.gz` | conjuntos de entrenamiento, con `is_synthetic` y `peso`. |
| `estimaciones_{ds}.csv` | la vuelta: coeficientes (por nombre) y métricas. |

---

## Qué se sube a git

| ruta | ¿en git? | por qué |
|---|---|---|
| `config.yaml`, `correr_todo.bat`, `python/`, `R/` | sí | es el código |
| `data/raw/` | sí, si tamaño y licencia lo permiten | sin esto no es replicable |
| `data/processed/`, `data/folds/` | **no** | regenerable con el paso 1 y 2 |
| `results/*.csv`, `results/*.xlsx` | sí | permite ver resultados sin correr nada |
| `logs/` | no | ruido |
| `requirements.txt` | sí | fija las versiones Python |

Si `data/raw/` es demasiado grande para git, usa git-lfs o publica los datos en
Zenodo y deja el DOI en `data/raw/README.md`. Para fijar las versiones de R,
inicializa `renv` (`renv::init()` + `renv::snapshot()`) y commitea `renv.lock`.

---

## Añadir un dataset

1. Deja el CSV en `data/raw/`.
2. Copia un bloque de `config.yaml` bajo `datasets:` y ajusta columnas,
   alternativas del modelo, tipos de variable y la lista `excluir` del CVAE.
3. `python python/scripts/00_verificar.py` para confirmar que las columnas
   existen.
4. `.\correr_todo.bat nuevo` (o el pipeline paso a paso).

No hay que tocar código en ninguno de los dos lenguajes.

---

## Notas de reproducibilidad

- Las semillas están en `config.yaml` (`semillas:`). Con la misma semilla, los
  folds y las muestras sintéticas salen idénticos.
- Cada fila de `estimaciones_{ds}.csv` identifica su origen con `metodo`,
  `repeticion` y `fold`, así que cualquier resultado se puede rehacer aislado.
- `R/03_estimar.R` guarda `sessionInfo()` en `logs/` en cada corrida.
- El pin `tensorflow<2.16` en `requirements.txt` es necesario: el código usa la
  API de Keras integrada en TF 2.10–2.15, que cambió en 2.16.