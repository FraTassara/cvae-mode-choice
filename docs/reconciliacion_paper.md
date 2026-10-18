# Reconciliación del paper publicado con el pipeline corregido

**Fuente de cada celda:** `docs/reconciliacion_paper.csv` (349 celdas). Lo genera
`python/scripts/15_reconciliacion.py` a partir de `docs/valores_paper.csv`
(valores transcritos de `docs/Paper_Aceptado_TRR_CVAE.pdf`, con su página) y de
`tablas_paper/` (commit `0fa7ba8`, `config_hash` `821d5b4010d2`). El detalle
técnico de cada causa está en `docs/diagnostico_corrigendum.md` (notas N1–N9,
§§ 2, 6, 10 y 11).

## Resumen por estado

| tabla | coincide | redondeo | procedencia | especificación | aritmética | total |
|---|---|---|---|---|---|---|
| 2 | 18 | 2 | — | — | — | 20 |
| 6 | 1 | — | — | — | 8 | 9 |
| 7 | 2 | — | 6 | 1 | — | 9 |
| 8 | 15 | — | 15 | 15 | — | 45 |
| 9 | — | — | 5 | — | — | 5 |
| 10 | 4 | — | 4 | 4 | — | 12 |
| 11 | 53 | 2 | 98 | 87 | — | 240 |
| texto | 3 | — | 4 | — | 2 | 9 |
| **total** | **96** | **4** | **132** | **107** | **10** | **349** |

**No queda ninguna celda `genuina`.** Las 14 celdas de la columna ORIG. de
LC-Centro que no coincidían pasaron primero a `pendiente_diagnostico` y,
tras el diagnóstico N10 (abajo), a `especificacion`.

- **procedencia:** el número publicado salió de otra versión de los resultados o
  de otra regla de agregación.
- **especificación:** cambió el modelo de Swissmetro.
- **aritmética:** una cuenta o un cociente del texto está mal hecho.
- **genuina:** ninguna de las causas anteriores explica la diferencia.

De las 53 coincidencias de la Tabla 11 (SM-Centro 24, LC-Centro 22,
Swissmetro 7), 26 son valores de magnitud ≤ 0,001 con tres decimales. Su
coincidencia no informa mucho. Las coincidencias de la Tabla 8 y la Tabla 10
de LC-Centro, en cambio, son exactas en todas las celdas.

## Qué cambia, tabla por tabla (lenguaje para el corrigendum)

**Tabla 2 (cuotas de mercado).** Sin cambios. Dos cuotas difieren en el
segundo decimal por redondeo (Metro SM-Centro: 32,46 %; Bus LC-Centro:
27,14 %).

**Tabla 6 (observaciones sintéticas).** Hay que reemplazarla completa
(`aritmetica`). La tabla publicada mezcla tamaños de la muestra completa con un
balanceo que en SM-Centro cuenta solo la clase minoritaria (279 = 332 − 53), y
en Swissmetro usa una submuestra de 9.036 observaciones que no es la estimada
(10.710; 9.036 son las filas con auto disponible). Con el procedimiento
descrito en el pie de la propia tabla (balanceo dentro de cada fold de
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

- **LC-Centro:** sin cambios.
- **SM-Centro:** cambia el promedio, no los modelos. La tabla publicada promediaba
  solo las particiones en que Apollo informó convergencia, y ese subconjunto era
  distinto por método (class-weighting: 11 de 50). Ahora entran las 50 en todos
  los métodos. El CVAE sigue siendo el mejor en F1 macro (0,727).
- **Swissmetro:** cambian todas las celdas por la re-especificación (abajo). El
  F1 macro sube en todos los métodos (original 0,441; CVAE 0,494; ROS 0,490;
  SMOTE 0,477; class-weighting 0,491). **El orden cambia:** el CVAE pasa de
  ser el peor de los métodos de balanceo a tener el F1 más alto, aunque con
  diferencias frente a ROS y class-weighting menores que la desviación entre
  particiones (≈ 0,009). Deja de valer la frase "SMOTE performs best".

**Tabla 9 (elasticidad propia del tiempo del auto, SM-Centro).** Se reemplaza
(`procedencia`). La tabla publicada provenía de un único modelo por método,
estimado en una partición. Promediada sobre las 50 particiones, la elasticidad
del CVAE es **−0,088** (SD 0,131): es negativa en media y positiva solo en el
30 % de las particiones. Los demás métodos: original −0,146, ROS −0,143, SMOTE
−0,153, class-weighting −0,143.

**La afirmación "under CVAE it becomes positive (+0.014)" no se sostiene y debe
retirarse.** Las elasticidades que el CVAE invierte de forma sistemática en
SM-Centro son las del tiempo de caminata (positiva en el 96 % de las
particiones) y del tiempo de espera (100 %).

**Tabla 10 (sensibilidad a la elipse).**

- **LC-Centro:** sin cambios.
- **SM-Centro:** varía en el tercer decimal por la misma regla de convergencia
  que la Tabla 8 (0,728 / 0,727 / 0,727 / 0,729).
- **Swissmetro:** 0,494 en los cuatro niveles.

La conclusión (robustez al nivel de la elipse) se mantiene.

**Tabla 11 (coeficientes).**

- **SM-Centro, columna ORIG.:** se reproduce con el MLE sobre la muestra
  completa (dos celdas difieren por redondeo).
- **SM-Centro y LC-Centro, columnas por método:** vienen de la versión
  sometida, que tenía fuga de datos, y se reemplazan por las del pipeline
  corregido.
- **Swissmetro:** cambia por la re-especificación.
- **LC-Centro, columna ORIG.: otra especificación (N10).** La columna publicada
  se estimó con la especificación de la Tabla 4, que incluye el ingreso en las
  utilidades de Car Driver, Car Passenger y Bus (`b_mi_cd`, `b_mi_cp`,
  `b_mi_bus`). La especificación del pipeline para LC-Centro no lo incluye.
  Con N = 398 y esos tres términos agregados se reproducen los 9 coeficientes
  publicados al tercer decimal, y 17 de 18 S.E. (el restante difiere por
  redondeo). Evidencia: `docs/diagnostico_lc_orig.csv`, generado por
  `experimental/diagnostico_lc_orig.R`.

  Las otras hipótesis no reproducen la columna:

  | variante | distancia media a lo publicado |
  |---|---|
  | sin disponibilidad | 0,749 |
  | N = 444 con el modo 3 | 0,348 |
  | N = 444 con el modo 3 e ingreso | 0,203 |
  | spec del pipeline | 0,167 |
  | ingreso + jefe de hogar en CP | 0,024 |
  | **ingreso (Tabla 4)** | **0,000** |

  **El paper es internamente inconsistente en LC-Centro.** La Tabla 4 y la
  columna ORIG. de la Tabla 11 usan el ingreso; las Tablas 8 y 10, que el
  pipeline reproduce exactas, se estimaron sin él. El corrigendum tiene que
  elegir una especificación para LC-Centro y declararla:
  - (a) corregir la Tabla 4 y la columna ORIG. a la especificación sin
    ingreso;
  - (b) re-estimar LC-Centro completo con ingreso. Esto cambia la
    especificación de LC, que hoy es un invariante del repo, y requiere
    autorización explícita.
- **SM-Centro, filas nuevas:** se agregan los coeficientes de ingreso
  (`b_mi_cd`, `b_mi_bus`), que están en la especificación (Tabla 4) pero se
  habían omitido. Se reportan por cada 10.000 de ILM.

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

## Hallazgos que el corrigendum debe declarar aunque no corresponden a una celda

1. **Re-especificación de Swissmetro (G1, §10).** El costo de tren y Swissmetro
   pasa a ser el costo efectivo, `CO·(GA = 0)` (convención de Biogeme), y el
   auto queda disponible solo con `CAR_AV = 1`. Es un cambio de modelo, no una
   corrección de replicación. La frase sobre Salas et al. (2022) se reemplaza
   por la atribución a Bierlaire et al. (2001) (N7). Hay borrador de nota a la
   Tabla 3 en el §10 del diagnóstico.
2. **"Dos de tres" se mantiene, pero no por las razones del paper (§11).** Con
   la especificación publicada de Swissmetro, `b_cost` salía positivo en
   **todos** los métodos por omitir GA (N8). Con la especificación completa,
   la inversión era exclusiva del CVAE (tres de tres). Con el costo efectivo,
   ningún método la invierte. La cifra coincide con la del abstract, pero el
   fundamento cambió y en SM-Centro uno de los coeficientes citados también
   (`b_wt` en lugar de `b_ivt`).
3. **Mecanismos propios del CVAE (N2, N6).** El CVAE genera las variables
   condicionales (disponibilidad del auto, abono GA) de forma independiente de
   los atributos que dependen de ellas. En Swissmetro, entre sus sintéticos que
   eligen tren:
   - ninguno quedaba sin auto, frente al 45 % de los reales;
   - la correlación entre GA y el costo de tren caía de 0,92 a 0,01.

   ROS y SMOTE conservan ambas relaciones. Con las dos declaradas como ceros
   estructurales, el CVAE recupera el signo de `b_cost`. Es un hallazgo
   metodológico que refuerza la recomendación práctica del paper: verificar la
   coherencia estructural de los datos sintéticos antes de estimar.
4. **Regla de convergencia.** Todas las cifras promedian las 50 particiones en
   todos los métodos y se informa cuántas no convergieron según Apollo. En
   SM-Centro, tras reescalar el ingreso, todas convergen.
