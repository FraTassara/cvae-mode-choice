# \# cvae-mode-choice

# 

# Aumento de datos con \*\*Conditional VAE\*\* para modelos de elección modal (MNL) con

# clases desbalanceadas. \*\*Python\*\* genera los datos sintéticos y calcula los

# diagnósticos; \*\*R + Apollo\*\* estima los modelos. Los dos lados se comunican por

# archivos con un contrato explícito, y todo se controla desde un único

# `config.yaml`.

# 

# El objetivo del proyecto es comparar métodos de balanceo (ROS, SMOTE, CVAE,

# ponderación por clase) frente a una línea base sin balancear, midiendo tanto el

# desempeño predictivo como la \*\*coherencia económica de los coeficientes\*\* — es

# decir, si los signos y magnitudes de los β siguen teniendo sentido después de

# aumentar los datos.

# 

# \---

# 

# \## El flujo en una imagen

# 

# ```

# config.yaml  ─────────────────────────────────────────────┐

# &#x20;  │  (fuente única de verdad: rutas, semillas, filtros,   │

# &#x20;  │   especificación de utilidades del MNL, tipos de      │

# &#x20;  │   variable y exclusiones del CVAE)                    │

# &#x20;  │                                                       │

# &#x20;  ▼                                                       ▼

# data/raw/\*.csv                                     R lee el MISMO

# &#x20;  │                                               config.yaml

# &#x20;  │  \[1] 01\_folds.py       normaliza + particiona         │

# &#x20;  ▼                                                       │

# data/processed/{ds}\_original.csv  ──────────────────────►  │

# data/folds/{ds}\_folds.csv  ─────────────────────────────►  │

# &#x20;  │                                                       │

# &#x20;  │  \[2] 02\_sinteticos.py  balanceo DENTRO de cada fold   │

# &#x20;  ▼                                                       │

# data/processed/{ds}\_{metodo}\_train.csv.gz  ─────────────►  │

# &#x20;  │                                                       ▼

# &#x20;  │                                           \[3] R/03\_estimar.R

# &#x20;  │  (verificación del contrato antes de estimar)      Apollo MNL

# &#x20;  │                                                       │

# &#x20;  │                                                       ▼

# &#x20;  │                                    results/estimaciones\_{ds}.csv

# &#x20;  │                                                       │

# &#x20;  │  \[4] 04\_reporte.py   →   \[5] 05\_excel.py              │

# &#x20;  ▼                                                       ▼

# results/tabla\_\*.csv               results/tablas\_{ds}.xlsx

# 

# Diagnósticos (independientes, al final):

# &#x20;  \[6] 06\_diagnosticos.py     JSD, gaps de atributos, ratios   (Python)

# &#x20;  \[7] 07\_hiperparametros.py  búsqueda de arquitectura del CVAE (Python)

# &#x20;  \[8] R/08\_efectos\_marginales.R  elasticidades por método     (R)

# ```

# 

# \---

# 

# \## Empezar

# 

# \### Requisitos

# 

# \- \*\*Python 3.11\*\* (no 3.12+: TensorFlow < 2.16 aún no tiene ruedas para ellas).

# \- \*\*R 4.x\*\* con Apollo, para el paso de estimación.

# 

# \### Instalación

# 

# ```bash

# git clone <url> cvae-mode-choice

# cd cvae-mode-choice

# 

# \# Python

# python -m venv .venv

# source .venv/bin/activate            # Windows: .venv\\Scripts\\activate.bat

# pip install -r requirements.txt

# 

# \# R (una vez)

# Rscript -e "install.packages(c('apollo','yaml','here','optparse','digest'))"

# ```

# 

# \### Datos

# 

# Copia los CSV originales en `data/raw/` con los nombres que declara

# `config.yaml`. Luego verifica que todo esté en su sitio:

# 

# ```bash

# python python/scripts/00\_verificar.py

# ```

# 

# Comprueba dependencias, presencia de los CSV y —lo más útil— que \*\*todas las

# columnas mencionadas en `config.yaml` existan de verdad\*\* en cada archivo. Una

# columna mal escrita se detecta aquí y no a los cuarenta minutos de estimación.

# 

# \### Correr todo

# 

# En Windows hay un script que encadena los pasos para un dataset:

# 

# ```bat

# .\\correr\_todo.bat sm\_centro --con-busqueda    REM incluye búsqueda de hiperparámetros

# .\\correr\_todo.bat sm\_centro                    REM el pipeline completo

# ```

# 

# En Mac/Linux, o si prefieres control paso a paso, ver "Pipeline detallado".

# 

# \---

# 

# \## Estructura

# 

# ```

# config.yaml              FUENTE ÚNICA DE VERDAD. La leen Python y R.

# correr\_todo.bat          Encadena el pipeline para un dataset (Windows).

# requirements.txt         Dependencias Python.

# 

# data/

# &#x20; raw/                   CSV originales.            \[en git si el tamaño lo permite]

# &#x20; processed/             Normalizados + sintéticos. \[gitignored, regenerable]

# &#x20; folds/                 Asignación de particiones. \[gitignored, regenerable]

# 

# python/

# &#x20; cvae/                  Paquete: CVAE, tipos de variable, muestreo latente.

# &#x20;   config.py            Dataclasses de configuración.

# &#x20;   config\_repo.py       Lector de config.yaml + manifiesto de reproducibilidad.

# &#x20;   variables.py         Tipos de variable y generadores condicionales.

# &#x20;   data.py              Carga, transformaciones, escalado.

# &#x20;   model.py             Encoder, decoder, pérdidas, KL annealing.

# &#x20;   latent.py            Elipses de confianza, muestreo por rechazo.

# &#x20;   generate.py          Muestreo, decodificación, re-ensamblado.

# &#x20;   diagnostics.py       JSD, benchmark interno, ratios, gaps de atributos.

# &#x20;   search.py            Random search estratificado sobre el ELBO.

# &#x20;   seeding.py           Semillas Python + NumPy + TensorFlow.

# &#x20;   plots.py             Espacio latente, distribuciones.

# &#x20;   pipeline.py          Puente entre config.yaml y el paquete.

# &#x20; scripts/

# &#x20;   00\_verificar.py      Chequeo de entorno y columnas.

# &#x20;   01\_folds.py          Paso 1: normalizar + particionar.

# &#x20;   02\_sinteticos.py     Paso 2: generar por partición.

# &#x20;   04\_reporte.py        Paso 4: tablas finales.

# &#x20;   05\_excel.py          Paso 5: exportar a Excel con formato.

# &#x20;   06\_diagnosticos.py   Diagnósticos de calidad de los sintéticos.

# &#x20;   07\_hiperparametros.py  Búsqueda de arquitectura del CVAE.

# &#x20;   tests\_contrato.py    Verifica las invariantes que R da por supuestas.

# 

# R/

# &#x20; config.R               Espejo de config\_repo.py. Lee el mismo YAML.

# &#x20; apollo\_modelo.R        Construye apollo\_probabilities desde config.yaml.

# &#x20; metricas.R             Métricas, matriz de confusión en orientación fija.

# &#x20; 03\_estimar.R           Paso 3: estimación (un script parametrizado).

# &#x20; 08\_efectos\_marginales.R  Elasticidades por método.

# 

# docs/CONTRATO.md         Especificación del contrato entre lenguajes.

# results/                 Tablas y diagnósticos finales.

# logs/                    Salida de Apollo, sessionInfo(). \[gitignored]

# ```

# 

# \---

# 

# \## Los métodos de balanceo

# 

# Cada método es una forma distinta de corregir el desbalance de clases. Se

# declaran en `config.yaml` bajo `metodos:`.

# 

# | método | qué hace | genera datos |

# |---|---|---|

# | `original` | nada; línea base para comparar | no |

# | `ROS` | duplica filas reales de las clases minoritarias | sí (copias) |

# | `SMOTE` | interpola entre vecinos cercanos de la clase minoritaria | sí (interpolación) |

# | `CVAE` | aprende la distribución y muestrea observaciones nuevas | sí (generativo) |

# | `class\_weights` | pondera la verosimilitud por el inverso de la frecuencia | no (pesos) |

# 

# El punto de comparación no es solo cuál predice mejor, sino cuál lo hace \*\*sin

# destruir la interpretación económica\*\* de los coeficientes.

# 

# \---

# 

# \## Las decisiones de diseño que hacen esto replicable

# 

# \### 1. Un solo `config.yaml`, leído por los dos lenguajes

# 

# Rutas, semillas, filtros, especificación de utilidades del MNL, tipos de

# variable y exclusiones del CVAE viven en un único archivo. Python lo lee con

# `yaml.safe\_load`, R con `yaml::read\_yaml`.

# 

# Consecuencia práctica: \*\*cambiar el modelo es editar YAML, no código\*\*.

# `R/apollo\_modelo.R` construye `apollo\_probabilities` por metaprogramación desde

# ese archivo, así que añadir un atributo a una alternativa son dos líneas en el

# config y ni Python ni R se tocan. Y un filtro (por ejemplo `AGE != 6` en

# Swissmetro) no puede quedar aplicado en un lado y no en el otro.

# 

# Regla del repo: si un script contiene una ruta, una semilla o un nombre de

# columna escrito a mano, es un bug.

# 

# \### 2. Rutas relativas, sin `setwd()`

# 

# Python resuelve con `pathlib` desde la raíz del repo; R con `here::here()`.

# Ninguna ruta absoluta, y nada de `setwd(getActiveDocumentContext()$path)`, que

# requiere RStudio interactivo e impide correr con `Rscript` — que es lo que

# necesita cualquiera que clone el repo.

# 

# \### 3. El sobremuestreo ocurre DENTRO de cada partición

# 

# Los folds se construyen sobre los datos \*\*originales\*\*. Para cada partición, el

# balanceo se aplica solo a las filas de entrenamiento, y la validación contiene

# únicamente filas reales.

# 

# Generar los sintéticos antes de partir en folds parece equivalente y no lo es:

# una fila sintética del entrenamiento se genera a partir de filas reales que

# pueden caer en validación, y el modelo terminaría evaluándose sobre datos que

# influyeron en su propio entrenamiento. Por eso, en el caso del CVAE, se entrena

# un modelo nuevo por partición. `tests\_contrato.py` verifica que ninguna fila de

# validación aparezca en su propio train.

# 

# \### 4. El CVAE modela solo las variables que corresponden

# 

# `config.yaml` declara, por dataset, una lista `excluir` con las columnas que el

# CVAE \*\*no\*\* debe modelar: identificadores, variables fuera del modelo y

# atributos de modos que no forman parte de la especificación. Esas columnas se

# remuestrean de la distribución empírica condicional al modo, así que siguen

# apareciendo en el CSV sintético con valores plausibles, pero no ocupan capacidad

# del espacio latente ni distorsionan la reconstrucción.

# 

# \### 5. Un manifiesto que solo salta cuando hace falta

# 

# `data/processed/MANIFEST.json` guarda un hash de la parte de `config.yaml` que

# afecta a los datos intermedios (rutas, filtros, clases, semilla y parámetros de

# validación). Editar la arquitectura del CVAE o la especificación del MNL \*\*no\*\*

# invalida los folds ya generados; cambiar las clases o la semilla, sí. Si hay

# desajuste, los scripts abortan con un mensaje claro en vez de mezclar datos de

# dos versiones.

# 

# \---

# 

# \## Correcciones metodológicas al CVAE

# 

# El generador incorpora varias correcciones respecto de la implementación inicial

# (detalladas en el documento de resumen). Las principales:

# 

# \- \*\*Muestreo estocástico en elipses de confianza\*\* en lugar de recorrer una

# &#x20; recta con `np.linspace`. El código anterior generaba puntos sobre un segmento

# &#x20; del plano latente (correlación ±1 exacta); ahora se ajusta una normal por clase

# &#x20; y se muestrea por rechazo dentro de la elipse al nivel de confianza indicado.

# \- \*\*Tratamiento por tipo de variable\*\*: continuas y zero-inflated (con

# &#x20; `log(x+1)`) pasan por el CVAE; binarias, empíricas y disponibilidades se

# &#x20; generan aparte condicionando al modo, con propagación de ceros estructurales.

# \- \*\*KL annealing\*\*: el peso del término KL sube de 0 a 1 durante las primeras

# &#x20; épocas, para estabilizar el entrenamiento.

# \- \*\*Semillas completas\*\* (Python, NumPy, TensorFlow) para reproducibilidad.

# \- \*\*Validación cruzada real\*\*: folds estratificados sobre datos originales,

# &#x20; balanceo solo en el train, evaluación sobre filas reales.

# 

# \---

# 

# \## Pipeline detallado (paso a paso)

# 

# Reemplaza `sm\_centro` por el dataset que quieras (`lc\_centro`, `whalen`,

# `swissmetro`).

# 

# ```bash

# \# 0. Verificar entorno y columnas

# python python/scripts/00\_verificar.py

# 

# \# 1. Normalizar y construir folds

# python python/scripts/01\_folds.py --dataset sm\_centro

# 

# \# (opcional) Buscar la arquitectura del CVAE. NO escribe en config.yaml:

# \# imprime el bloque para copiar y guarda results/hiperparametros\_sm\_centro.csv.

# python python/scripts/07\_hiperparametros.py --dataset sm\_centro

# 

# \# 2. Generar conjuntos de entrenamiento (todos los métodos del config)

# python python/scripts/02\_sinteticos.py --dataset sm\_centro

# 

# \# 2b. Verificar el contrato ANTES de estimar

# python python/scripts/tests\_contrato.py --dataset sm\_centro

# 

# \# 3. Estimar en R con Apollo (acumula: no borra métodos ya estimados)

# Rscript R/03\_estimar.R --dataset sm\_centro

# 

# \# 4-5. Tablas y Excel

# python python/scripts/04\_reporte.py --dataset sm\_centro

# python python/scripts/05\_excel.py --dataset sm\_centro

# 

# \# 6, 8. Diagnósticos

# python python/scripts/06\_diagnosticos.py --dataset sm\_centro

# Rscript R/08\_efectos\_marginales.R --dataset sm\_centro

# ```

# 

# Para estimar un solo método (por ejemplo tras cambiar la arquitectura del CVAE),

# `03\_estimar.R` acepta `--metodo CVAE` y \*\*reemplaza solo esas filas\*\*,

# conservando los demás métodos ya estimados. Después hay que rehacer los pasos 4

# y 5, que consolidan todos los métodos.

# 

# \### Análisis de sensibilidad de elipses

# 

# El nivel de confianza de la elipse se puede sobrescribir por línea de comandos;

# cada nivel se guarda como un método aparte:

# 

# ```bash

# python python/scripts/02\_sinteticos.py --dataset sm\_centro --metodo CVAE --confianza 0.70

# Rscript R/03\_estimar.R --dataset sm\_centro --metodo CVAE\_conf70

# ```

# 

# \---

# 

# \## Qué produce

# 

# En `results/`, por dataset:

# 

# \- `tablas\_{ds}.xlsx` — tres pestañas: \*\*Predictiva\*\* (accuracy, F1, log-

# &#x20; verosimilitud, VOT por método), \*\*Coeficientes\*\* (media, S.D. entre

# &#x20; particiones, S.E. del estimador) y \*\*Cambios de signo\*\* (coeficientes que

# &#x20; invierten el signo esperado, con resaltado).

# \- `estimaciones\_{ds}.csv` — una fila por (método, repetición, fold), materia

# &#x20; prima de las tablas.

# \- `jsd\_{ds}.csv`, `sobremuestreo\_{ds}.csv`, `gaps\_atributos\_{ds}.csv` — los

# &#x20; diagnósticos. El de \*\*gaps\*\* es el que explica el mecanismo: si la diferencia

# &#x20; de atributos entre alternativa elegida y no elegidas se distorsiona en los

# &#x20; datos sintéticos, ahí está el origen de las reversiones de signo del MNL.

# \- `hiperparametros\_{ds}.csv` — la tabla de intentos de la búsqueda, si se corrió.

# \- `efectos\_marginales\_{ds}.csv` — elasticidades por método.

# 

# \---

# 

# \## El contrato Python ↔ R

# 

# Cuatro archivos cruzan la frontera entre lenguajes; fuera de ellos y de

# `config.yaml`, los dos lados no comparten nada. Está documentado en

# `docs/CONTRATO.md` y verificado por `tests\_contrato.py`. En resumen:

# 

# | archivo | rol |

# |---|---|

# | `{ds}\_original.csv` | dataset ya filtrado, con `row\_id`. R lee este, no el crudo. |

# | `{ds}\_folds.csv` | asignación de particiones (`row\_id`, `repeticion`, `fold`, `split`). |

# | `{ds}\_{metodo}\_train.csv.gz` | conjuntos de entrenamiento, con `is\_synthetic` y `peso`. |

# | `estimaciones\_{ds}.csv` | la vuelta: coeficientes (por nombre) y métricas. |

# 

# \---

# 

# \## Qué se sube a git

# 

# | ruta | ¿en git? | por qué |

# |---|---|---|

# | `config.yaml`, `correr\_todo.bat`, `python/`, `R/` | sí | es el código |

# | `data/raw/` | sí, si tamaño y licencia lo permiten | sin esto no es replicable |

# | `data/processed/`, `data/folds/` | \*\*no\*\* | regenerable con el paso 1 y 2 |

# | `results/\*.csv`, `results/\*.xlsx` | sí | permite ver resultados sin correr nada |

# | `logs/` | no | ruido |

# | `requirements.txt` | sí | fija las versiones Python |

# 

# Si `data/raw/` es demasiado grande para git, usa git-lfs o publica los datos en

# Zenodo y deja el DOI en `data/raw/README.md`. Para fijar las versiones de R,

# inicializa `renv` (`renv::init()` + `renv::snapshot()`) y commitea `renv.lock`.

# 

# \---

# 

# \## Añadir un dataset

# 

# 1\. Deja el CSV en `data/raw/`.

# 2\. Copia un bloque de `config.yaml` bajo `datasets:` y ajusta columnas,

# &#x20;  alternativas del modelo, tipos de variable y la lista `excluir` del CVAE.

# 3\. `python python/scripts/00\_verificar.py` para confirmar que las columnas

# &#x20;  existen.

# 4\. `.\\correr\_todo.bat nuevo` (o el pipeline paso a paso).

# 

# No hay que tocar código en ninguno de los dos lenguajes.

# 

# \---

# 

# \## Notas de reproducibilidad

# 

# \- Las semillas están en `config.yaml` (`semillas:`). Con la misma semilla, los

# &#x20; folds y las muestras sintéticas salen idénticos.

# \- Cada fila de `estimaciones\_{ds}.csv` identifica su origen con `metodo`,

# &#x20; `repeticion` y `fold`, así que cualquier resultado se puede rehacer aislado.

# \- `R/03\_estimar.R` guarda `sessionInfo()` en `logs/` en cada corrida.

# \- El pin `tensorflow<2.16` en `requirements.txt` es necesario: el código usa la

# &#x20; API de Keras integrada en TF 2.10–2.15, que cambió en 2.16.

