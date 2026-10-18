# Diagnóstico para el corrigendum (Fase 0)

Fecha: 2026-10-17 · commit base: `0d385b0` · `config_hash` del MANIFEST: `8d47b2392530`

Solo lectura: no se modificó ningún archivo del pipeline. Los scripts de
verificación usados están en el scratchpad de la sesión; los comandos clave se
reproducen abajo.

## 0. Contradicciones entre el encargo y el repo (el repo manda)

| # | Encargo | Repo | Consecuencia |
|---|---|---|---|
| C1 | `docs/Paper_Aceptado_TRR_CVAE.pdf` | **No existe.** `docs/` solo contiene `CONTRATO.md`. | La Fase 3 (reconciliación) queda bloqueada hasta que se agregue el PDF. |
| C2 | "False convergence en 22–84 % de los folds en SM-Centro **y LC-Centro**" | LC-Centro converge **50/50 en todos los métodos**. Su especificación **no incluye ILM** (ni `b_mi_*`). En SM-Centro el rango es **16–78 %**. | El bug 1 solo afecta a SM-Centro. LC-Centro no necesita re-estimarse por ILM. |
| C3 | Swissmetro "10.728 crudo − `AGE != 6` = 10.710" | `data/raw/swissmetro.csv` tiene **10.719 filas y 12 columnas** (sin `CAR_AV`, `TRAIN_AV`, `SM_AV`, `PURPOSE`…). Ya viene sin `AGE = 6`: el filtro `AGE != 6` elimina **0** filas. Las 9 filas que se pierden son `CHOICE = 0`, eliminadas por `clases: [1, 2, 3]`. | El filtro de config es inocuo pero correcto. Hay que documentar que el CSV crudo es un extracto ya filtrado de la base de Biogeme (10.728 − 9 con AGE=6 = 10.719). |
| C4 | Reescalar ILM "en la preparación de datos" | SMOTE (`02_sinteticos.py::m_smote`) usa distancias euclídeas **sin escalar**; con ILM ≈ 42.800 (sd 33.125) ILM domina la búsqueda de vecinos. Reescalar en `01_folds` cambiaría los vecinos de SMOTE (y sus sintéticos), cambiaría el `config_hash` e invalidaría todos los `*_train.csv.gz` (incluido el CVAE, que es el paso caro). | Recomiendo reescalar **en la capa de estimación** (ver §1). Si prefieres la preparación de datos, se puede, pero SM-Centro/SMOTE dejará de ser comparable con lo validado. |

## 1. ILM: dónde se prepara y cómo reescalarlo

**Dónde vive hoy.** ILM no se transforma en ningún punto:

- `01_folds.py::normalizar` copia el crudo tal cual a `sm_centro_original.csv`
  (ILM: media 42.825, sd 33.125, rango 5.000–300.000).
- En el CVAE, ILM no está declarado en `cvae.variables` → tipo `CONTINUOUS`
  por defecto → entra al CVAE con `MinMaxScaler` (invariante a escala).
- En el MNL entra como `b_mi_cd: ILM` (car_driver) y `b_mi_bus: ILM` (bus)
  vía `R/apollo_modelo.R::construir_probabilities`. Con ILM ~1e4 frente a
  tiempos ~1e2 y costos ~1e1, el Hessiano queda mal condicionado.

**Propuesta (config-driven, sin tocar la especificación de utilidades).**

1. `config.yaml`, bloque `sm_centro.modelo`, clave nueva:
   ```yaml
   escalas: {ILM: 1.0e-4}   # ILM en decenas de miles; b_mi_* = efecto por cada 10.000 de ILM
   ```
   No está en `_config_de_datos()`, así que no cambia el `config_hash` ni
   invalida folds ni sintéticos (correcto: no los afecta).
2. `R/apollo_modelo.R`: función `aplicar_escalas(datos, spec)` que multiplica
   las columnas de `spec$escalas`; llamada al inicio de `estimar_mnl()` y de
   `predecir()`. Así la reciben automáticamente `03_estimar.R` y
   `08_efectos_marginales.R` (que perturba en +1 % de la media → elasticidades
   invariantes a escala).
3. Las utilidades no cambian: `b_mi_cd * ILM` sigue escrito igual; solo cambia
   la unidad de ILM y de `b_mi_*` (×1e4 respecto de lo validado).
4. Re-estimar SM-Centro en una etiqueta nueva (ver §5) para no pisar lo validado.

**Chequeo de sanidad esperado.** Al ser una reparametrización lineal, el
óptimo es el mismo: accuracy/F1/loglik prácticamente idénticos; `b_mi_*` ×1e4;
SE de `b_mi_*` ×1e4 y los demás SE más estables; convergencia ≈ 50/50.

## 2. `convergio` en `04_reporte.py` (V02)

`04_reporte.py` (líneas 101–107) **excluye** las filas con `convergio == FALSE`
antes de construir las tres tablas, y solo imprime un aviso global; ninguna
tabla registra cuántas particiones entran por método (salvo
`tabla_coeficientes.n_particiones`). `10_sensibilidad_elipses.py` (l. 51–52)
aplica el mismo filtro.

`convergio` = `isTRUE(modelo$successfulEstimation)` (`R/metricas.R:89`):
Apollo marca "false convergence" aunque el loglik sea el mismo.

Particiones convergidas (de 50) en `results/estimaciones_*.csv`:

| método | sm_centro | lc_centro | swissmetro | whalen |
|---|---|---|---|---|
| original | 35 | 50 | 50 | 49 |
| ROS | 32 | 50 | 50 | 49 |
| SMOTE | 42 | 50 | 50 | 50 |
| CVAE | 39 | 50 | 50 | 50 |
| CVAE_conf70 / 90 / 95 | 30 / 39 / 37 | 50 | 50 | — |
| class_weights | **11** | 50 | 50 | 48 |

Confirmado: en SM-Centro `class_weights` se promedia sobre 11/50 folds, y cada
método sobre un subconjunto distinto de particiones. Las métricas de los folds
no convergidos no son peores (p. ej. original: accuracy 0,832 no conv. vs 0,816
conv.), lo que confirma que el problema es numérico, no de ajuste.

**Regla propuesta:** *incluir todas las particiones* en las tablas (mismo
conjunto de 50 para todos los métodos, comparación pareada) y añadir columnas
`n_particiones` y `n_no_convergidas` por método en `tabla_predictiva` y
`tabla_coeficientes`. Solo se excluye una partición si la estimación **falló**
(error / SE no finitos), y eso se reporta. Tras el bug 1, la diferencia debería
ser casi nula en SM-Centro.

## 3. Verificaciones

### V01 — ¿De dónde sale N = 9.036 en Swissmetro?

```bash
.venv/Scripts/python.exe -c "import pandas as pd; df=pd.read_csv('data/raw/swissmetro.csv',sep=';'); s=df.query('CAR_TT>0 and CHOICE!=0'); print(len(s), s.CHOICE.value_counts().sort_index().to_dict())"
```

| filtro sobre el crudo del repo | N | train / sm / car |
|---|---|---|
| ninguno | 10.719 | (9 con CHOICE=0) 1.414 / 6.216 / 3.080 |
| `CHOICE != 0` (= pipeline) | **10.710** | 1.414 / 6.216 / 3.080 |
| `CAR_TT > 0 and CHOICE != 0` | **9.036** | 779 / 5.177 / 3.080 |

**Resultado:** 9.036 = muestra del pipeline restringida a individuos con auto
disponible. `CAR_AV` no está en el CSV del repo, pero `CAR_TT == 0` (y
`CAR_CO == 0`) identifica exactamente a esas 1.674 filas, que es como Biogeme
codifica `CAR_AV = 0`. Es un error de **aritmética/procedencia** del texto: la
Tabla 2 (10.710) sí corresponde al pipeline.

**Hallazgo colateral (no lo corrijo sin tu OK):** la especificación de
Swissmetro tiene `avail: null` en las tres alternativas, es decir, el auto se
trata como disponible en esas 1.674 filas con tiempo 0 y costo 0. Eso puede
sesgar `b_time` y `b_cost` (un "auto gratis e instantáneo" que nadie elige).
Lo correcto sería una disponibilidad derivada (`CAR_TT > 0`). Esto **cambia la
especificación** de Swissmetro, que me dijiste que ya está corregida; lo dejo
como decisión tuya.

### V02 — Política de `convergio`

Ver §2.

### V03 — Inversión de `b_cost` bajo CVAE en Swissmetro

```bash
cat results/tabla_signos_swissmetro.csv
```

Confirmado: `b_cost` invertido en el **100 %** de las particiones en CVAE,
CVAE_conf70, CVAE_conf90 y CVAE_conf95 (media +0,0003); **0 %** en original,
ROS, SMOTE y class_weights. `b_time` no se invierte en ningún método.

### V04 — Tabla 6 por fold

Conteos desde `data/processed/{ds}_{metodo}_train.csv.gz`, media sobre las 50
particiones (ROS, SMOTE y CVAE dan cifras idénticas: todos balancean a la
mayoritaria de cada fold).

| dataset | clase | reales/fold | sintéticos/fold | fracción sintética |
|---|---|---|---|---|
| sm_centro | 1 car driver | 42,4 | 223,2 | **84,0 %** |
| sm_centro | 4 metro | 148,0 | 117,6 | 44,3 % |
| sm_centro | 5 bus | 265,6 | 0 | 0 % |
| lc_centro | 2 car passenger | 32,8 | 67,2 | 67,2 % |
| lc_centro | 5 bus | 86,4 | 13,6 | 13,6 % |
| lc_centro | 4 metro | 99,2 | 0,8 | 0,8 % |
| swissmetro | 1 train | 1.131,2 | 3.841,6 | **77,3 %** |
| swissmetro | 3 car | 2.464,0 | 2.508,8 | 50,5 % |
| whalen | 1 cycling | 38,4 | 530,4 | 93,2 % |
| whalen | 4 car | 224,8 | 344,0 | 60,5 % |
| whalen | 3 hsr | 268,8 | 300,0 | 52,7 % |

Sintéticos totales por fold: SM 340,8 · LC 81,6 · SW 6.350,4 · Whalen 1.174,4.

**Origen probable del "81 %":** 223,2 / (223,2 + **53**) = 80,8 %, es decir,
sintéticos por fold divididos por (sintéticos + reales de la **muestra
completa**). Mezcla de denominadores → error de **aritmética**.

## 4. Bug 4 — Agregación del benchmark JSD

Cómo lo define `06_diagnosticos.py` (coherente): para cada variable y cada
modo, `JSD(real del modo, sintético del modo)` contra el **p95 del benchmark
del mismo modo y la misma variable**; `supera_benchmark` = supera en algún modo.

Problemas en `13_extraer_pendientes.py` (l. 132–147) y aguas arriba:

1. `jsd_benchmark_ref_p95` = media de **todos** los p95 (todas las variables ×
   todos los modos) y `jsd_sintetico_medio` = media de los JSD. Comparar dos
   promedios no es la prueba que define `06`: una variable con p95 alto
   compensa a otra con p95 bajo. Además en Swissmetro el "ref" (0,0075) es
   mucho menor que en SM (0,19) por tamaño muestral, así que el número
   agregado no es comparable entre datasets.
2. **`vars_sobre_benchmark = 17` de 29, no de 13.** `06` calcula JSD sobre
   **todas** las columnas de `{ds}_original.csv`: incluye `ID`, `HTS`, `JEFH`
   y los atributos de los modos 2 y 3 (`TVIA2`, `TESP3`, `DISP2`…), que están
   en `cvae.excluir` y no forman parte del MNL de SM-Centro. Las variables
   pertinentes son las que efectivamente se generan: las 13 declaradas en
   `cvae.variables` + las continuas no declaradas (`ILM`, `CTOT1_w`, `CTOT4_w`,
   `CTOT5_w`) = **17**. Sobre ese conjunto, SMOTE supera el benchmark en
   **8 de 17** (AUTLIC, TVIA1, TVIA4, TCAM1, TESP4, TESP5, DISP1, DISP4);
   el CVAE, en 12 de 17 (sus 12 excesos ya caían todos en variables
   pertinentes) y ROS, en 0 de 17.
3. Advertencias que dejo documentadas (no propongo cambiarlas): el JSD
   sintético usa solo la partición (0,0); el benchmark compara mitades de n/2
   reales, mientras el sintético compara ~223 sintéticos con 53 reales, así que
   los tamaños muestrales no coinciden.

**Propuesta:** en `06`, restringir las variables evaluadas a
`cols_datos − cvae.excluir − {id_col, choice_col}` (config-driven) y escribir
`n_vars` explícito; en `13`, reportar por método `vars_sobre_benchmark /
n_vars` y la **mediana del cociente JSD / p95** por variable-modo (cociente < 1
⇔ dentro del ruido), en vez de comparar promedios.

## 5. Plan de ediciones (Fase 1, pendiente de tu OK)

Cada edición se mostrará como diff antes de aplicarla.

| # | Archivo | Cambio |
|---|---|---|
| E1 | `config.yaml` | `sm_centro.modelo.escalas: {ILM: 1.0e-4}` con comentario de la unidad de `b_mi_*`. Opcional: `proyecto.etiqueta_resultados` para escribir en `results/corrigendum/` sin pisar lo validado. |
| E2 | `R/apollo_modelo.R` | `aplicar_escalas()`; llamada en `estimar_mnl()` y `predecir()`. |
| E3 | `R/config.R`, `python/cvae/config_repo.py` | Si se acepta la etiqueta de resultados: `ruta_estimaciones()` y `rutas.results` respetan la subcarpeta. |
| E4 | `python/scripts/04_reporte.py` | Regla única: incluir todas las particiones; columnas `n_particiones`, `n_no_convergidas` por método. Excluir solo estimaciones fallidas, reportándolo. |
| E5 | `python/scripts/10_sensibilidad_elipses.py` | Misma regla que E4. |
| E6 | `python/scripts/06_diagnosticos.py` | Variables evaluadas = generadas (config-driven), `n_vars` explícito. |
| E7 | `python/scripts/13_extraer_pendientes.py` | Benchmark coherente (§4) y `vars_sobre_benchmark / n_vars`. |
| E8 | nuevo `python/scripts/tabla_sinteticos.py` o dentro de `14_tablas_paper.py` | Tabla 6 por fold desde `*_train.csv.gz` (§3, V04). |
| E9 | `python/cvae/config_repo.py::escribir_manifiesto` | Hoy sobrescribe `paso_01` con solo el último dataset procesado (el MANIFEST actual solo lista swissmetro). Fusionar con el contenido previo para que la procedencia de la Fase 2 sea completa. |
| E10 | `CLAUDE.md` (nuevo) | Invariantes del repo. |

Orden de re-ejecución que te pasaré en la Fase 1 (sin `make all`): re-estimar
solo SM-Centro (todos los métodos, incluidas las variantes `CVAE_conf*`),
luego `04_reporte` para los 4 datasets, `06` para los 3 datasets con JSD, `08`
y `10` para SM-Centro, y `13`. Ningún paso de Python requiere regenerar
sintéticos con la opción recomendada en §1.

Otros detalles detectados de paso: `R/08_efectos_marginales.R` y
`python/scripts/08_efectos_marginales.R` son copias idénticas (conviene dejar
una sola); `make prueba` reescribe `results/estimaciones_sm_centro.csv` con
1 repetición sobre los resultados validados (acumula por método, pero
reemplaza las filas de `original` y `ROS`), así que para validar conviene
correrlo con la etiqueta de resultados nueva.

---

# Adenda (Fase 1, antes de `make prueba`)

## 6. D3 — Disponibilidad del auto en Swissmetro (PROPUESTA, sin implementar)

**Situación.** `swissmetro.modelo` tiene `avail: null` en las tres
alternativas. En 1.674 de las 10.710 filas el auto no está disponible, pero el
modelo lo trata como disponible con `CAR_TT = 0` y `CAR_CO = 0`: es un "auto
gratis e instantáneo" que nunca se elige. En los datos reales `CAR_TT == 0` y
`CAR_CO == 0` coinciden exactamente (0 filas con uno y no el otro), y es la
codificación de Biogeme para `CAR_AV = 0`. Derivar la disponibilidad no quita
filas: **N = 10.710**.

**Cómo generan hoy CAR_TT los métodos** (sintéticos de las 50 particiones):

| clase sintética | reales sin auto | ROS | SMOTE | CVAE |
|---|---|---|---|---|
| 1 train | 44,9 % | 45,0 % | 40,8 % (+0,7 % con 0 < CAR_TT < 5) | **0,0 %** |
| 3 car | 0 % | 0 % | 0 % | 0 % |

El CVAE trata `CAR_TT` y `CAR_CO` como `CONTINUOUS` y no hay una variable de
disponibilidad que genere los ceros estructurales. Por eso **todos** los
sintéticos que eligen tren tienen un auto disponible con tiempo y costo
positivos. **Hipótesis a verificar:** esta pérdida de la masa en cero es
candidata a explicar la inversión de `b_cost` en el 100 % de las particiones
CVAE (y en ningún otro método).

**Dos opciones para las filas sintéticas:**

| | (A) Derivada de CAR_TT | (B) Heredada (recomendada) |
|---|---|---|
| Implementación | `avail: "CAR_TT > 0"` en la alternativa car. `construir_probabilities` ya evalúa `avail` con `eval(parse())`, así que basta con editar el config. | `01_folds` deriva una columna `CAR_AV = (CAR_TT > 0)` (declarada en el config). En `cvae.variables` va `CAR_AV: {tipo: availability, depende: [CAR_TT, CAR_CO]}` y en la alternativa car `avail: CAR_AV`. |
| Reales | Exacta | Exacta |
| ROS | Exacta (copia filas) | Exacta (copia filas) |
| SMOTE | ~exacta. El 0,7 % interpolado entre un vecino sin auto y uno con auto queda "disponible" con tiempo casi nulo. | Interpola `CAR_AV` (valores fraccionarios). Hace falta una regla: redondear y propagar ceros a `CAR_TT`/`CAR_CO`. |
| CVAE | **Mala.** Con 0 % de ceros, todos los sintéticos tendrían auto disponible y el artefacto del CVAE seguiría intacto. | Bernoulli por clase (tren: 55 % con auto) más propagación de ceros estructurales. Es el mismo mecanismo que ya usan `DISP*` en SM y LC. |
| Costo | Solo re-estimar Swissmetro. No cambia el hash ni los sintéticos. | Cambia `_config_de_datos` → nuevo hash → regenerar **todos** los sintéticos de Swissmetro (CVAE y `CVAE_conf70/90/95` incluidos) y re-estimar. |

**Recomendación: (B).** Es la única opción en la que el CVAE reproduce que el
45 % de quienes eligen tren no tenía auto. Con (A), la comparación
CVAE-vs-resto mediría en parte el artefacto de la masa en cero y no el método.

**Detalle de SMOTE en (B).** La regla de redondeo y propagación sería genérica
para las columnas `availability`. Pero si se aplica a todos los datasets,
cambia también los sintéticos SMOTE de SM-Centro y LC-Centro, que hoy
interpolan `DISP*`. Propongo activarla por dataset en el config
(`smote_redondear_disponibilidad: true`, solo en Swissmetro) y dejar
documentado el mismo defecto en SM y LC.

**Impacto esperado.**

- Tabla 2 sin cambios (N y cuotas iguales). Tabla 6 sin cambios en conteos:
  el balanceo no depende de `CAR_AV`.
- Métodos con datos reales (original, class_weights, ROS): desaparece el
  auto fantasma en 1.674 filas → P(car) = 0 ahí y mejor log-likelihood. Es
  probable que suban la accuracy y el F1. Se espera que `b_time`/`b_cost` se
  alejen de cero: hoy el auto a costo cero que nunca se elige los atenúa. Cambian
  `asc_car` y el VTT.
- CVAE: es el punto decisivo. Si la inversión de `b_cost` desaparece, el
  corrigendum no puede decir "tres de tres datasets" y hay que reescribir esa
  afirmación. Si persiste, queda probado que no venía de la disponibilidad.
- Cambian las Tablas 7 (JSD de `CAR_TT`/`CAR_CO`, más la nueva `CAR_AV`), 8,
  9, 10 y 11 y la tabla de signos de Swissmetro.
- Todo se escribe en `results/corrigendum/`.

## 7. Tabla 7 — Columnas excluidas del CVAE en los `*_train.csv.gz`

**No son placeholders.** No hay NaN ni −1 (comprobado en la partición 0 de SM y
LC, para ROS, SMOTE y CVAE). Cada método trata estas columnas así:

- CVAE: `pipeline._schema_desde_yaml` las marca `EMPIRICAL` → se remuestrean
  **columna por columna** de la distribución empírica del modo. El 100 % de los
  valores existe en los reales, pero se pierde la coherencia dentro de la fila
  (p. ej. `DISP2 = 0` con `TVIA2 > 0`).
- ROS: copia de filas reales.
- SMOTE: interpolación, como cualquier otra columna (ID interpolado, etc.).

**Efecto en el JSD antiguo** (media de `jsd_modo_medio`, todas las columnas
frente a solo las generadas):

| dataset | método | n vars antes → ahora | JSD antes | JSD ahora | sesgo del antiguo | vars sobre benchmark antes → ahora |
|---|---|---|---|---|---|---|
| sm_centro | CVAE | 29 → 17 | 0,214 | 0,347 | **−38 %** | 12 → 12 |
| sm_centro | ROS | 29 → 17 | 0,026 | 0,027 | −6 % | 0 → 0 |
| sm_centro | SMOTE | 29 → 17 | 0,127 | 0,118 | **+8 %** | **17 → 8** |
| lc_centro | CVAE | 29 → 21 | 0,423 | 0,481 | −12 % | 27 → 19 |
| lc_centro | ROS | 29 → 21 | 0,290 | 0,285 | +2 % | 26 → 18 |
| lc_centro | SMOTE | 29 → 21 | 0,310 | 0,280 | +11 % | 28 → 20 |
| swissmetro | CVAE | 11 → 10 | 0,335 | 0,368 | −9 % | 6 → 6 |
| swissmetro | SMOTE | 11 → 10 | 0,0078 | 0,0082 | −6 % | 4 → 4 |
| whalen | CVAE | 38 → 37 | 0,555 | 0,570 | −3 % | 29 → 29 |

**Conclusión para el corrigendum.** Las columnas excluidas no inflaban el JSD
de forma pareja. Al remuestrearlas de la empírica, su JSD bajo el CVAE es casi
nulo (0,025 en SM), así que **favorecían al CVAE**: su JSD publicado está
subestimado hasta un 38 %. Bajo SMOTE, en cambio, lo inflaban y casi duplicaban
el conteo de variables sobre el benchmark. La corrección **empeora** el JSD del
CVAE en relación con SMOTE.

## 8. E9 y limitaciones

**E9 aplicado.** `Config.escribir_manifiesto` (`python/cvae/config_repo.py`)
ahora fusiona `paso_01` con el manifiesto previo cuando el `config_hash`
coincide, en vez de sobrescribirlo. Si el hash cambió, descarta lo previo.

**LIMITACIÓN (documentada, sin corregir): SMOTE usa distancias sin escalar.**
`02_sinteticos.py::m_smote` llama a `imblearn.SMOTE` sobre las columnas en sus
unidades originales. En SM-Centro, ILM (media 42.825, sd 33.125) tiene una
varianza varios órdenes de magnitud mayor que tiempos (sd ≈ 350) y costos
(sd ≈ 19). Los k vecinos de SMOTE quedan determinados casi solo por ILM, y la
interpolación mezcla filas parecidas en ingreso pero no en atributos del viaje.
Lo mismo ocurre en Swissmetro con `TRAIN_CO`/`SM_CO` (hasta 5.040/6.720)
frente a `AGE` o `GA`. No se corrige porque cambiaría los sintéticos SMOTE
validados de todos los datasets. Se reporta como limitación del baseline
SMOTE: es una implementación estándar sin estandarizar, que lo deja en
desventaja frente al CVAE (cuyo `MinMaxScaler` sí normaliza).

## 9. Notas para la reconciliación (Fase 3)

Estas notas se convertirán en filas de `docs/reconciliacion_paper.csv`.

**N1. Tabla 7: todas las celdas cambian, `procedencia`.** El JSD publicado
promediaba columnas que el método no genera (§7). Razones de JSD entre
métodos, antes (todas las columnas) → ahora (solo las generadas):

| dataset | CVAE/ROS | CVAE/SMOTE |
|---|---|---|
| sm_centro | 8,36 → **12,83** | 1,69 → **2,96** |
| lc_centro | 1,46 → **1,69** | 1,36 → **1,72** |
| whalen | 59,0 → **62,2** | 6,86 → **6,95** |
| swissmetro | 391,2 → 464,7 (D3, costo crudo) → **385,9** (D3 + G1) | 43,2 → 37,9 → **36,5** |

En Swissmetro la cifra nueva sale de los sintéticos regenerados con D3: son 11
variables, con `CAR_AV` incluida. La diferencia mezcla `procedencia` (sin
`ID`) y `especificacion` (D3).

**N6. Hallazgo propio de mecanismo, no presente en el paper: el CVAE rompía
el acoplamiento GA–costo en Swissmetro.** En el crudo, quien tiene abono anual
(GA = 1) registra en `TRAIN_CO`/`SM_CO` el precio del abono (1.286–5.040 /
1.715–6.720 CHF). Quien no lo tiene registra 4–576 / 6–768 CHF. GA y costo
están perfectamente acoplados: no hay ninguna fila con GA = 0 y costo ≥ 1.286.
Medido sobre los sintéticos que eligen tren (corrida "costo crudo", las 50
particiones):

| | reales | ROS | SMOTE | CVAE |
|---|---|---|---|---|
| corr(GA, TRAIN_CO) | 0,92 | 0,92 | 0,92 | **0,01** |
| GA = 0 con costo de abono | 0 % | 0 % | 0 % | **23,8 %** |
| GA = 1 con costo ordinario | 0 % | 0 % | 0 % | **25,2 %** |

El CVAE muestrea GA (binaria, externa) por clase, sin relación con el
`TRAIN_CO` que genera. Así, casi un cuarto de sus sintéticos de tren "paga"
≈ 2.600 CHF sin abono y aun así elige tren. Es el mismo tipo de defecto que
N2 (disponibilidad 45 % → 0 %) y es candidato directo a explicar la inversión
de `b_cost`. La corrida G1 (§10) lo pone a prueba.

**N7. Atribución incorrecta, `procedencia`.** Para Swissmetro el paper dice
"utilities identical to those used by Salas et al. (2022)". Ese paper no
estima Swissmetro (sus datasets son Toronto–Montreal, Las Condes, San Miguel y
McMaster). La frase debe corregirse en el corrigendum. La especificación de
Swissmetro tiene que atribuirse a la convención de Biogeme (§10) y declararse
como propia.

**N8. Hipótesis de mecanismo (CONFIRMADA por la corrida G1, §11): sesgo por
variable omitida vía los portadores de abono.** En la spec reducida publicada
(solo ASC, `b_time` y `b_cost`), GA no está en la utilidad. Los portadores de
abono registran costos de tren de 1.286–5.040 CHF y eligen tren en mayor
proporción: el 44,6 % de quienes eligen tren tiene GA, frente al 13,4 % en
Swissmetro y el 1,7 % en auto. Al omitir GA, el costo queda asociado
positivamente a elegir tren, y `b_cost` sale positivo en el **100 % de las
particiones de todos los métodos** (corrida A de §11). Con GA en la utilidad
(corridas B y C), el baseline y el sobremuestreo recuperan el signo y solo
falla el CVAE. Eso es coherente con que el CVAE sea el único método que rompe
el acoplamiento GA–costo (N6): en sus sintéticos, GA ya no identifica a los
portadores del costo alto, y el término `b_ga` no puede absorber ese efecto.
Predicción que G1 pone a prueba: con el costo efectivo y GA acoplada por
`anula_si`, el CVAE debería recuperar el signo de `b_cost`.

**N9. La Tabla 9 (elasticidades) sale de UNA partición, `procedencia`.**
`08_efectos_marginales.R` estima un solo modelo por método, en la partición
`--repeticion 0 --fold 0` (valores por defecto). En SM-Centro, el CVAE da una
elasticidad propia de TVIA **positiva** (0,008–0,021) porque en esa partición
`b_ivt = +0,00017`. Sobre las 50 particiones, `b_ivt` del CVAE no está invertido
en media (−0,0010; invertido en el 30 %). Los coeficientes que el CVAE invierte
de forma exclusiva en SM-Centro son `b_wkt` (96 %) y `b_wt` (100 %). En
LC-Centro, la elasticidad positiva sí es representativa (`b_ivt` invertido
en el 90 % de las particiones). **Corregido en la Fase 2:** `08` estima ahora
en las 50 particiones (`efectos_marginales_resumen_{ds}.csv`). En SM-Centro,
la elasticidad propia de TVIA1 (Car Driver) bajo el CVAE es **−0,088 (SD
0,131)**, positiva en el **30 %** de las particiones. El +0,014 publicado no se
reproduce. Las demás métricas: original −0,146 (8 %), ROS −0,143 (2 %), SMOTE
−0,153 (2 %), class-weighting −0,143 (2 %). Las elasticidades propias que el
CVAE invierte de forma sistemática en SM-Centro son las de caminata (TCAM1,
positiva en el 96 %) y espera (TESP4 y TESP5, positivas en el 100 %),
coherentes con `b_wkt` y `b_wt`. Fuente: `tablas_paper/tabla_09.csv`.

**N10. La columna ORIG. de LC-Centro en la Tabla 11 usa otra
especificación, `especificacion`.** La reconciliación marcó primero 14 celdas
como `genuina` y luego como `pendiente_diagnostico`: la especificación de LC
no cambió en el corrigendum, así que una columna entera distinta apuntaba a
otra muestra u otra especificación. Diagnóstico, en orden:

1. *¿N = 398?* Sí. La Tabla 2 del paper suma 125 + 41 + 124 + 108 = 398, igual
   que el pipeline. El crudo tiene 444 filas, con 46 del modo 3 que
   `clases: [1, 2, 4, 5]` excluye. Estimar con N = 444 y el modo 3 no reproduce
   la columna (distancia media a lo publicado 0,348; con ingreso, 0,203).
2. *¿Ingreso?* Sí, exacto. Con `b_mi_cd`, `b_mi_cp` y `b_mi_bus` (ingreso en
   Car Driver, Car Passenger y Bus, como dice la Tabla 4) se reproducen los 9
   coeficientes publicados al tercer decimal y 17 de 18 S.E.; el S.E. de
   `b_wkt` difiere por redondeo. Agregar también jefe de hogar en CP empeora
   el ajuste (0,024). La diferencia de costo (−0,003 publicado frente a
   −0,014) se explica porque el ingreso absorbe parte del efecto de
   `CTOT_w`; no hace falta otra definición de `CTOT_w`.
3. *¿Scripts antiguos en git?* No. `git log --all --diff-filter=D -- '*.R'`
   no devuelve nada: los 13 `.R` que reemplazó `03_estimar.R` nunca
   estuvieron en este repositorio, y la especificación de LC en `config.yaml`
   es idéntica en todos los commits.

Evidencia: `experimental/diagnostico_lc_orig.R` → `docs/diagnostico_lc_orig.csv`.

**Consecuencia: el paper es internamente inconsistente en LC-Centro.** La
Tabla 4 y la columna ORIG. de la Tabla 11 incluyen el ingreso. Las Tablas 8 y
10, que el pipeline reproduce exactas, se estimaron sin él. La decisión
(corregir la Tabla 4, o re-estimar LC con ingreso, que rompe el invariante
"no modificar la spec de LC") es de la autora.

**N5. [REVOCADA por §11] La inversión de `b_cost` bajo el CVAE persiste con D3.** Ocurre en el
100 % de las particiones de CVAE y de `CVAE_conf70/90/95` (media +0,0001,
|t| medio 3,05) y en el 0 % de los demás métodos. Queda descartada la
hipótesis de §6 (masa en cero de `CAR_TT`). El corrigendum debe decir "tres de
tres datasets", no "dos". **REVOCADA por §11**: con G1, el CVAE no invierte
`b_cost` en ninguna partición y el conteo vuelve a "dos de tres" (SM-Centro y
LC-Centro).

## 10. G1 — Costo efectivo en Swissmetro (RE-ESPECIFICACIÓN)

**Definición (convención de Biogeme):**

```
TRAIN_COST = TRAIN_CO · (GA == 0)
SM_COST    = SM_CO    · (GA == 0)
```

Quien tiene abono anual paga 0 por el viaje marginal en tren y Swissmetro. El
crudo registra para esas filas el precio del abono, que no es el costo de
la elección. `CAR_CO` no cambia.

**Implementación (config-driven).**

- `datasets.swissmetro.derivadas` crea `TRAIN_COST` y `SM_COST` en
  `01_folds.py`.
- En `modelo.alternativas`, `b_cost` pasa a usar `TRAIN_COST` (train) y
  `SM_COST` (sm). `b_ga` sigue en train y sm.
- En el CVAE: `TRAIN_CO` y `SM_CO` pasan a `cvae.excluir` (no los usa el MNL)
  y `GA: {tipo: binary, depende: [TRAIN_COST, SM_COST], anula_si: 1}`.
  `propagate_structural_zeros` ahora atiende cualquier variable con
  `depende` y su `anula_si` (por defecto 0, como las disponibilidades: SM y LC
  no cambian).
- SMOTE: la misma regla declarativa (`02_sinteticos._ceros_estructurales`).
- Partición 0: GA = 1 ⇒ costo 0 en el 100 % de los sintéticos de los tres
  métodos; ninguna fila con GA = 0 y costo 0.

**Clasificación: RE-ESPECIFICACIÓN** (resuelta por la autora el 2026-10-18).
Salas et al. (2022) no estima Swissmetro: sus datasets son Toronto–Montreal,
Las Condes, San Miguel y McMaster. Por lo tanto, G1 no corrige una replicación
fallida: es un cambio de modelo, y el corrigendum debe declararlo como tal.
La justificación es la convención del dataset en Biogeme (Bierlaire et al.,
2001): para quien tiene abono anual, el costo marginal del viaje en tren y en
Swissmetro es cero. Todas las celdas de Swissmetro (Tablas 8–11 y signos)
cambian respecto de la corrida "costo crudo".

**Borrador de la nota a la Tabla 3 (en inglés, como el paper):**

> *Swissmetro.* The car alternative is available only to respondents with a
> car (CAR_AV = 1, identified by CAR_TT > 0), and the travel cost of train and
> Swissmetro enters the utility as the effective cost TRAIN_COST =
> TRAIN_CO·(GA = 0) and SM_COST = SM_CO·(GA = 0), following the convention for
> this dataset in Biogeme (Bierlaire et al., 2001). Holders of an annual season
> ticket (GA) face a zero marginal cost for these modes, because the raw cost
> fields record the price of the season ticket; this specification replaces
> the one used in the originally published version of this paper.

**Trazabilidad.** La corrida anterior (D3 con costo crudo) está guardada en
`results/corrigendum_sw_costocrudo/`, junto con el `config_usado.yaml` con el
que salió. La corrida G1 está en `results/corrigendum/`.

**N2. Hallazgo propio, no presente en el paper: el CVAE borraba la no
disponibilidad del auto en Swissmetro.** Entre los sintéticos que eligen tren,
la fracción sin auto (`CAR_TT = CAR_CO = 0`) era 45,0 % en ROS y 40,8 % en
SMOTE (reales: 44,9 %), y **0,0 % en el CVAE**. El CVAE modelaba `CAR_TT` y
`CAR_CO` como continuas sin una variable de disponibilidad: todos sus
sintéticos de tren tenían un auto disponible con tiempo y costo positivos.
Con D3 (`CAR_AV` heredada, declarada como `availability`), la partición 0
queda en 45,4 % para el CVAE.

**N3. La fila SMOTE de Swissmetro difiere por DOS causas.** Hay que
anotarlas juntas en cada celda SMOTE de Swissmetro (Tablas 7–11 y signos):
(i) la disponibilidad `CAR_AV` en la especificación y en los datos, y (ii)
la regla nueva de ceros estructurales para SMOTE (redondeo de la
disponibilidad interpolada y propagación a `CAR_TT`/`CAR_CO`). Esa regla es
declarativa: lee `cvae.variables.*.tipo = availability` y `depende`, igual que
el CVAE. Solo se activa en datasets con `acoplar_disponibilidad: true` (hoy,
Swissmetro). Efecto observado en la partición 0: 52,9 % de sintéticos de tren
sin auto, frente al 45,5 % real. El exceso viene del redondeo sobre vecinos
elegidos con distancias sin escalar (§8).

**N4. Swissmetro original, ROS y class_weights: `especificacion`.** Sus
diferencias se deben solo a `avail: CAR_AV` en la alternativa car.

## 11. Swissmetro: comparación de las cuatro corridas

Fuente: `docs/comparacion_corridas_swissmetro.csv`, una fila por corrida y
método, con la columna `archivo_fuente`. Política de `convergio`: entran todas
las particiones (50 por método en las cuatro corridas). Las variantes
`CVAE_conf70/90/95` dan lo mismo que el CVAE base hasta el tercer decimal y se
omiten aquí; están en el CSV.

| corrida | especificación | fuente |
|---|---|---|
| A | reducida publicada: ASC + `b_time` + `b_cost` | git `3b90f01` |
| B | completa (`b_ga`, `b_seats`, `b_age`, `b_luggage`), sin D3 | git `0d385b0` (`results/`) |
| C | B + D3 (`CAR_AV`), costo crudo | `results/corrigendum_sw_costocrudo/` |
| D | C + G1 (costo efectivo, GA → ceros estructurales) | `results/corrigendum/` |

**% de particiones con `b_cost` invertido, por método:**

| corrida | original | ROS | SMOTE | class_weights | CVAE |
|---|---|---|---|---|---|
| A | 100 % | 100 % | 100 % | 100 % | 100 % |
| B | 0 % | 0 % | 0 % | 0 % | **100 %** |
| C | 0 % | 0 % | 0 % | 0 % | **100 %** |
| D | 0 % | 0 % | 0 % | 0 % | **0 %** |

**`b_cost` (media):**

| corrida | original | ROS | SMOTE | class_weights | CVAE |
|---|---|---|---|---|---|
| A | +0,0003 | +0,0004 | +0,0004 | +0,0004 | +0,0008 |
| B | −0,0012 | −0,0013 | −0,0013 | −0,0012 | +0,0003 |
| C | −0,0012 | −0,0013 | −0,0013 | −0,0013 | +0,0001 |
| D | −0,0065 | −0,0060 | −0,0066 | −0,0061 | −0,0084 |

**`b_time` (media):**

| corrida | original | ROS | SMOTE | class_weights | CVAE |
|---|---|---|---|---|---|
| A | −0,0041 | −0,0033 | −0,0036 | −0,0033 | −0,0068 |
| B | −0,0051 | −0,0043 | −0,0051 | −0,0043 | −0,0074 |
| C | −0,0119 | −0,0125 | −0,0140 | −0,0125 | −0,0149 |
| D | −0,0130 | −0,0138 | −0,0157 | −0,0138 | −0,0159 |

**F1 macro (media):**

| corrida | original | ROS | SMOTE | class_weights | CVAE |
|---|---|---|---|---|---|
| A | 0,252 | 0,335 | 0,342 | 0,335 | 0,332 |
| B | 0,394 | 0,494 | 0,498 | 0,494 | 0,387 |
| C | 0,484 | 0,518 | 0,510 | 0,518 | 0,466 |
| D | 0,441 | 0,490 | 0,477 | 0,491 | **0,494** |

**Lectura.**

1. **N8 confirmada.** Sin GA en la utilidad (A), `b_cost` sale positivo en
   todos los métodos: es sesgo por variable omitida, no un efecto del CVAE.
   Con GA (B, C), solo el CVAE invierte, porque es el único que rompe el
   acoplamiento GA–costo (N6). Con el costo efectivo y GA acoplada (D), el
   CVAE recupera el signo en el 100 % de las particiones (|t| medio 12,6) y
   **ningún método invierte `b_cost`**.
2. **N5 queda revocada.** Con la especificación D, el CVAE invierte signos de
   forma exclusiva en **dos de tres** datasets (SM-Centro: `b_wkt`, `b_wt`;
   LC-Centro: `b_ivt`, `b_tcw`). Es lo que dice el paper. Con la
   especificación C serían tres. El conteo depende de la especificación de
   Swissmetro, y el corrigendum debe declararlo junto con la re-especificación
   G1.
3. **Cambia el orden predictivo.** En C, el orden de F1 es CVAE < original <
   SMOTE < ROS ≈ class_weights. En D, el CVAE pasa a tener el F1 más alto
   (0,494, frente a 0,491 de class_weights y 0,441 del original). Las
   diferencias entre CVAE, class_weights y ROS son menores que 0,005; la
   desviación entre particiones es ≈ 0,009. En accuracy, el original sigue
   primero en todas las corridas (0,654 en D).
4. **El VTT pasa a ser interpretable en todos los métodos** en D: 1,90–2,40,
   en las unidades de `b_time`/`b_cost` (CHF por minuto). En C, el del CVAE
   era ≈ −144.
5. **Cómo cambió la evidencia contra el CVAE en Swissmetro con cada
   especificación.** Se cuentan los datasets donde el CVAE invierte un signo y
   ningún otro método lo hace:

   | corrida | ¿Swissmetro cuenta contra el CVAE? | conteo |
   |---|---|---|
   | A (publicada) | no: `b_cost` invertido en **todos** los métodos | 2 |
   | B (+ GA, seats, age, luggage) | sí: solo el CVAE invierte `b_cost` | **3** |
   | C (+ D3) | sí | **3** |
   | D (+ G1) | no: **ningún** método invierte `b_cost` | 2 |

   Corregir la especificación **no redujo la evidencia en cada paso**: el
   paso a la especificación completa (B) la **aumentó**, al hacer la inversión
   exclusiva del CVAE; D3 no la cambió, y **solo G1 la redujo de 3 a 2**.
6. **"Dos de tres" coincide con el abstract publicado, pero por razones
   distintas.** En la corrida publicada (A), Swissmetro quedaba fuera del
   conteo porque la inversión de `b_cost` era general a todos los métodos:
   un sesgo por omitir GA (N8), no una propiedad del CVAE. En D queda fuera
   porque ningún método invierte el signo. La cifra es la misma, pero el
   fundamento no: en A, la ausencia de Swissmetro no era evidencia a favor
   del CVAE, porque el modelo estaba mal especificado para todos los métodos.
   Con el PDF ya disponible: el paper atribuye al CVAE en SM-Centro la
   inversión de `b_wkt` y `b_ivt`. En las 50 particiones, los coeficientes
   invertidos son `b_wkt` (96 %) y `b_wt` (100 %); `b_ivt` lo está solo en el
   30 % (N9). En LC-Centro coincide: `b_ivt` y `b_tcw`. El conteo "dos de
   tres" se mantiene, pero uno de los coeficientes citados para SM-Centro
   cambia (detalle en la Fase 3).
