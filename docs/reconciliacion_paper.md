# Reconciliación del paper publicado con el pipeline corregido

**Fuente de cada celda:** `docs/reconciliacion_paper.csv` (353 celdas). Lo genera
`python/scripts/15_reconciliacion.py` a partir de `docs/valores_paper.csv`
(valores transcritos de `docs/Paper_Aceptado_TRR_CVAE.pdf`, con su página) y de
`tablas_paper/` (pie: commit de origen `3fdbd87`, `config_hash` `821d5b4010d2`).
El detalle técnico de cada causa está en `docs/diagnostico_corrigendum.md`
(notas N1–N10 y N13, §§ 2, 6, 10 y 11).

Especificaciones usadas en el corrigendum:

- **SM-Centro:** la publicada, con ingreso reescalado (por cada 10.000 de ILM).
- **LC-Centro:** la de la Tabla 4 del paper, con ingreso en CD, CP y Bus y sin
  jefe de hogar (N10). Es una corrección a lo declarado.
- **Swissmetro:** re-especificado con D3 y G1 (§10).

## Resumen por estado

| tabla | coincide | redondeo | procedencia | especificación | aritmética | total |
|---|---|---|---|---|---|---|
| 2 | 18 | 2 | — | — | — | 20 |
| 6 | 1 | — | — | — | 8 | 9 |
| 7 | 2 | — | 6 | 1 | — | 9 |
| 8 | — | — | 30 | 15 | — | 45 |
| 9 | — | — | 5 | — | — | 5 |
| 10 | — | — | 8 | 4 | — | 12 |
| 11 | 67 | 3 | 97 | 73 | — | 240 |
| texto | 3 | — | 6 | 2 | 2 | 13 |
| **total** | **91** | **5** | **152** | **95** | **10** | **353** |

**No queda ninguna celda `genuina`.**

- **procedencia:** el número publicado salió de otra versión de los resultados,
  de otra regla de agregación o de otra especificación que la declarada en el
  paper.
- **especificación:** el corrigendum cambia el modelo o lo corrige a lo
  declarado (Swissmetro; afirmaciones que dependen de ello).
- **aritmética:** una cuenta o un cociente del texto está mal hecho.

De las 67 coincidencias de la Tabla 11 (SM-Centro 24, LC-Centro 36,
Swissmetro 7), 17 son la columna ORIG. de LC-Centro reproducida con ingreso;
la celda restante de esa columna difiere por redondeo. Muchas de las demás son
valores de magnitud ≤ 0,001 con tres decimales, cuya coincidencia informa
poco.

## Qué cambia, tabla por tabla (lenguaje para el corrigendum)

**Tabla 2 (cuotas de mercado).** Sin cambios. Dos cuotas difieren en el
segundo decimal por redondeo (Metro SM-Centro: 32,46 %; Bus LC-Centro:
27,14 %).

**Tabla 4 (utilidades de LC-Centro y SM-Centro).** Se retira el jefe de hogar
(HH). La Tabla 4 lo declara, pero la estimación publicada no lo incluía: con HH
la columna ORIG. publicada no se reproduce (N10).

**Tabla 6 (observaciones sintéticas).** Hay que reemplazarla completa
(`aritmetica`). La tabla publicada mezcla tamaños de la muestra completa con un
balanceo que en SM-Centro cuenta solo la clase minoritaria (279 = 332 − 53), y
en Swissmetro usa una submuestra de 9.036 observaciones que no es la estimada
(10.710; 9.036 son las filas con auto disponible). Con el procedimiento que
describe el pie de la propia tabla (balanceo dentro de cada fold de
entrenamiento), los valores por fold son:

- SM-Centro: 456 reales, 340,8 sintéticos (razón 0,75).
- LC-Centro: 318,4 reales, 81,6 sintéticos (razón 0,26).
- Swissmetro: 8.568 reales, 6.350,4 sintéticos (razón 0,74).

En consecuencia, el "81 %" de sintéticos en Car Driver de SM-Centro (texto,
pp. 11 y 13) debe decir **84 %**: 223 sintéticos frente a 42 reales por fold.

**Tabla 7 (JSD).** Cambian 7 de 9 celdas. El JSD publicado promediaba también
columnas que ningún método genera (identificadores y atributos fuera del
modelo). Esas columnas rebajaban el JSD del CVAE e inflaban el de SMOTE (N1).
Sobre las variables generadas, el CVAE diverge más de lo informado:

| | ROS | SMOTE | CVAE |
|---|---|---|---|
| SM-Centro | 0,027 | 0,118 | 0,347 |
| LC-Centro | 0,285 | 0,280 | 0,481 |
| Swissmetro | 0,001 | 0,008 | 0,305 |

Las frases "8 veces mayor en SM-Centro" y "335 veces mayor en Swissmetro" pasan
a **13** y **384**.

**Tabla 8 (desempeño predictivo).**

- **SM-Centro:** cambia el promedio, no los modelos. La tabla publicada
  promediaba solo las particiones en que Apollo informó convergencia, y ese
  subconjunto era distinto por método (class-weighting: 11 de 50). Ahora entran
  las 50 en todos los métodos. El CVAE sigue siendo el mejor en F1 macro
  (0,727).
- **LC-Centro (N13):** la tabla publicada se estimó **sin ingreso**. Se
  reproduce exacta desde `results/corrigendum_lc_sin_ingreso/`. La columna
  ORIG. de la Tabla 11, en cambio, se estimó con ingreso. Con la especificación
  de la Tabla 4:
  - F1 macro: original 0,667; CVAE 0,678; SMOTE 0,709; ROS 0,710;
    class-weighting 0,710.
  - **El CVAE pasa a superar al original**, pero sigue siendo el peor de los
    métodos de balanceo.
  - Deja de valer "below the unbalanced baseline (0.676 vs. 0.682)".
- **Swissmetro:** cambian todas las celdas por la re-especificación.
  - F1 macro: original 0,441; CVAE 0,494; ROS 0,490; SMOTE 0,477;
    class-weighting 0,491.
  - El CVAE pasa de ser el peor de los métodos de balanceo a tener el F1 más
    alto, con diferencias frente a ROS y class-weighting menores que la
    desviación entre particiones (≈ 0,009).
  - Deja de valer "SMOTE performs best".

En conjunto, **el CVAE mejora el F1 del original en los tres datasets** (el
paper dice dos) y tiene el F1 más alto en SM-Centro y en Swissmetro (el paper
dice solo SM-Centro). En Swissmetro la ventaja no es distinguible del ruido
entre particiones.

**Tabla 9 (elasticidades, SM-Centro).** Se reemplaza (`procedencia`). Las
**cinco** celdas publicadas, una por método, provenían de un único modelo
estimado en una partición. Por ejemplo, el original pasa de −0,180 a −0,146
sobre 50 particiones. La tabla corregida reporta media ± SD y % de particiones
positivas:

| método | espera Metro | espera Bus | viaje auto |
|---|---|---|---|
| Original | −0,119 ± 0,028 (0 %) | −0,063 ± 0,014 (0 %) | −0,146 ± 0,076 (8 %) |
| CVAE | **+0,116 ± 0,020 (100 %)** | **+0,062 ± 0,011 (100 %)** | −0,088 ± 0,131 (30 %) |
| ROS | −0,052 ± 0,036 (6 %) | −0,027 ± 0,018 (6 %) | −0,143 ± 0,054 (2 %) |
| SMOTE | −0,084 ± 0,034 (0 %) | −0,047 ± 0,019 (0 %) | −0,153 ± 0,070 (2 %) |
| Class-weighting | −0,056 ± 0,033 (0 %) | −0,028 ± 0,017 (0 %) | −0,143 ± 0,055 (2 %) |

Hay un párrafo en inglés para el corrigendum en
`docs/parrafo_elasticidades_corrigendum.md`. Correcciones de texto:

- "walking time (b wkt) and travel time (b ivt)" → **"waiting time and walking
  time (b_wt, b_wkt)"**.
- La frase sobre un "systematic pattern across observations" en el tiempo de
  viaje del auto se reemplaza por la evidencia entre particiones: la
  elasticidad propia de la **espera** es positiva en el 100 % de las
  particiones, y la del tiempo de viaje solo en el 30 %.

**Tabla 10 (sensibilidad a la elipse).**

- **SM-Centro:** varía en el tercer decimal por la regla de convergencia
  (0,728 / 0,727 / 0,727 / 0,729).
- **LC-Centro:** 0,680 / 0,678 / 0,678 / 0,679. Cambia por la especificación
  con ingreso (N13).
- **Swissmetro:** 0,494 en los cuatro niveles.

La conclusión (robustez al nivel de la elipse) se mantiene.

**Tabla 11 (coeficientes).**

- **Columna ORIG., SM-Centro y LC-Centro:** se reproduce con el MLE sobre la
  muestra completa (LC-Centro ya con ingreso). Solo hay diferencias de
  redondeo.
- **Columnas por método, SM-Centro y LC-Centro:** vienen de la versión
  sometida, que tenía fuga de datos, y se reemplazan por las del pipeline
  corregido.
- **Swissmetro:** cambia por la re-especificación.
- **Filas nuevas de ingreso:** en SM-Centro (`b_mi_cd`, `b_mi_bus`) y
  LC-Centro (`b_mi_cd`, `b_mi_cp`, `b_mi_bus`). Están en la especificación
  (Tabla 4), pero la Tabla 11 publicada las omitía: con ILM sin escalar sus
  coeficientes son del orden de 1e-5 y redondean a 0,000. Se reportan por cada
  10.000 de ILM.

## Afirmaciones del texto

| afirmación | publicado | pipeline | estado |
|---|---|---|---|
| Sintéticos en Car Driver, SM-Centro | 81 % | 84 % | aritmética |
| Muestra de Swissmetro | 9.036 | 10.710 | aritmética |
| JSD CVAE / ROS, SM-Centro | 8× | 13× | procedencia |
| JSD CVAE / ROS, Swissmetro | 335× | 384× | procedencia |
| Inversiones exclusivas del CVAE, SM-Centro | `b_ivt`, `b_wkt` | `b_wkt`, `b_wt` | procedencia |
| Inversiones exclusivas del CVAE, LC-Centro | `b_ivt`, `b_tcw` | `b_ivt`, `b_tcw` | coincide |
| `b_cost` negativo en todos los métodos, Swissmetro | sí | sí | coincide |
| Datasets con inversión exclusiva del CVAE | 2 de 3 | 2 de 3 | coincide |
| Fuente de las utilidades de Swissmetro | Salas et al. (2022) | convención Biogeme | procedencia |
| HH en la utilidad de LC-Centro | sí (Tabla 4) | no | procedencia |
| Efecto sistemático del CVAE, SM-Centro | tiempo de viaje del auto | espera Metro y Bus | procedencia |
| Datasets donde el CVAE mejora el F1 del original | 2 | 3 | especificación |
| Datasets donde el CVAE tiene el mayor F1 | SM-Centro | SM-Centro; Swissmetro | especificación |

## Hallazgos que el corrigendum debe declarar aunque no corresponden a una celda

1. **Re-especificación de Swissmetro (G1, §10).** El costo de tren y Swissmetro
   pasa a ser el costo efectivo, `CO·(GA = 0)` (convención de Biogeme), y el
   auto queda disponible solo con `CAR_AV = 1`. Es un cambio de modelo, no una
   corrección de replicación. La frase sobre Salas et al. (2022) se reemplaza
   por la atribución a Bierlaire et al. (2001) (N7). Hay borrador de nota a la
   Tabla 3 en el §10 del diagnóstico.
2. **Especificaciones mezcladas en el paper publicado (N13).** En LC-Centro, la
   Tabla 8 (CV) se estimó sin ingreso y la columna ORIG. de la Tabla 11 con
   ingreso. En Swissmetro, la Tabla 8 corresponde a la especificación reducida
   (corrida A) y la Tabla 11 a la completa (corrida B). El corrigendum usa una
   sola especificación por dataset en todas las tablas.
3. **"Dos de tres" se mantiene, pero no por las razones del paper (§11).** Con
   la especificación publicada de Swissmetro, `b_cost` salía positivo en
   **todos** los métodos por omitir GA (N8). Con la especificación completa,
   la inversión era exclusiva del CVAE (tres de tres). Con el costo efectivo,
   ningún método la invierte. La cifra coincide con la del abstract, pero el
   fundamento cambió, y en SM-Centro uno de los coeficientes citados también
   (`b_wt` en lugar de `b_ivt`). En LC-Centro, con ingreso, se mantienen `b_ivt`
   y `b_tcw` como inversiones exclusivas del CVAE, y `b_wt` heredado del dato.
4. **Mecanismos propios del CVAE (N2, N6).** El CVAE genera las variables
   condicionales (disponibilidad del auto, abono GA) de forma independiente de
   los atributos que dependen de ellas. En Swissmetro, entre sus sintéticos que
   eligen tren:
   - ninguno quedaba sin auto, frente al 45 % de los reales;
   - la correlación entre GA y el costo de tren caía de 0,92 a 0,01.

   ROS y SMOTE conservan ambas relaciones. Con las dos declaradas como ceros
   estructurales, el CVAE recupera el signo de `b_cost`. Es un hallazgo
   metodológico que refuerza la recomendación práctica del paper: verificar la
   coherencia estructural de los datos sintéticos antes de estimar.
5. **Regla de convergencia.** Todas las cifras promedian las 50 particiones en
   todos los métodos y se informa cuántas no convergieron según Apollo. En
   SM-Centro y LC-Centro, con el ingreso reescalado, todas convergen.
