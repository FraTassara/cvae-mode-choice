# cvae-mode-choice — invariantes del repo

Pipeline replicable Python + R que compara métodos de balanceo (original, ROS,
SMOTE, CVAE, class_weights) sobre un MNL estimado con Apollo, en cuatro
datasets (sm_centro, lc_centro, swissmetro, whalen). Se trabaja en español.

## Fuente de verdad

- `config.yaml` es la ÚNICA fuente de verdad: rutas, semillas, filtros,
  especificación del MNL (bloque `modelo:` por dataset) y variables del CVAE
  (bloque `cvae:`). Una ruta, semilla o nombre de columna hard-codeado en un
  script es un bug.
- `R/apollo_modelo.R` construye las utilidades desde el config por
  metaprogramación. La especificación del MNL NO se escribe a mano en ningún
  script.
- `modelo.escalas` (p. ej. `{ILM: 1.0e-4}` en sm_centro) reescala covariables
  solo dentro de `estimar_mnl()` / `predecir()`. Cambia la unidad del
  coeficiente, no la utilidad ni los datos de `data/processed/`.

## Flujo

`01_folds` → `02_sinteticos` → `03_estimar` → `04_reporte`, orquestado por el
`Makefile` (`make all`, `make DATASET=x all`, `make prueba`).
`data/processed/MANIFEST.json` guarda el hash de la parte del config que afecta
a folds y datos (`Config._config_de_datos`) e invalida lo generado si cambia.
`make all` lo corre el usuario, no el asistente.

## Contrato de archivos

- `{ds}_original.csv` (con `row_id`), `{ds}_folds.csv` (row_id, repeticion,
  fold, split), `{ds}_{metodo}_train.csv.gz` (con `is_synthetic`, `peso`).
- `estimaciones_{ds}.csv`: formato ANCHO, una fila por
  metodo/repeticion/fold, columnas `beta_<param>`, `se_<param>`, `convergio`,
  `vot` y métricas. `03_estimar.R` ACUMULA por método: solo reemplaza las filas
  de los métodos que re-estima.
- `label_map` en los datasets es `{id_de_modo: clase_codificada}` (se invierte
  para mapear clase → modo).

## Reglas de análisis

- Política de `convergio` (`python/cvae/convergencia.py`): entran todas las
  particiones estimadas, igual para todos los métodos; se reportan
  `n_particiones` y `n_no_convergidas`. Solo se descartan estimaciones sin
  coeficientes finitos.
- Fracciones de sintéticos: numerador y denominador del MISMO fold
  (`diagnostics.oversampling_por_fold`).
- Ceros estructurales: toda variable de `cvae.variables` con `depende` anula
  esas columnas cuando vale `anula_si` (0 por defecto, p. ej. `CAR_AV`; 1 para
  `GA` en Swissmetro). Los aplican el CVAE (`propagate_structural_zeros`) y,
  con `acoplar_disponibilidad: true`, SMOTE; `02_sinteticos.py` aborta si un
  método deja una fila sin acoplar.
- Columnas derivadas (`datasets.<ds>.derivadas`) se calculan en `01_folds.py`
  y entran en el hash del MANIFEST.
- JSD: solo variables generadas (sin `id_col` ni `cvae.excluir`); cada
  (variable, modo) se compara con el p95 del benchmark de esa variable y ese
  modo.

## Qué no tocar

- NO modificar la especificación de SM-Centro ni de LC-Centro.
- La de Swissmetro ya fue corregida (base `asc_train`; `b_age`, `b_luggage`
  en car; `b_ga` en train y sm; `b_seats` en sm) y re-estimada, incluidas
  `CVAE_conf70/90/95`.
- Los resultados nuevos van con etiqueta/carpeta distinta a los validados
  (`proyecto.rutas.results`); nunca sobrescribir resultados validados sin dejar
  rastro.
- Antes de editar un archivo, mostrar el diff al usuario. Si el repo contradice
  una instrucción, decirlo en vez de forzarla.
