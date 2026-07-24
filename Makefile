# ============================================================================ #
#  Makefile — punto de entrada único.
#
#  `make` sirve como documentación ejecutable: las dependencias entre pasos
#  están escritas, así que se ve de un vistazo qué produce qué. Y si cambias
#  config.yaml, make sabe qué hay que rehacer.
#
#      make setup        instalar dependencias (Python + R)
#      make datos        paso 1: normalizar + folds
#      make sinteticos   paso 2: generar conjuntos de entrenamiento
#      make estimar      paso 3: Apollo en R
#      make reporte      paso 4: tablas finales
#      make all          todo, para todos los datasets
#      make DATASET=lc_centro all    todo, para uno solo
#      make prueba       corrida rápida (1 repetición) para verificar el flujo
#      make limpiar      borrar intermedios
# ============================================================================ #

DATASETS ?= sm_centro lc_centro whalen swissmetro
DATASET  ?=
METODOS  ?= original ROS SMOTE CVAE

PY := python
RS := Rscript

# Si se pasa DATASET=..., se opera solo sobre ese.
ifneq ($(DATASET),)
  OBJETIVOS := $(DATASET)
else
  OBJETIVOS := $(DATASETS)
endif

.PHONY: all setup datos sinteticos estimar reporte prueba limpiar limpiar-todo verificar ayuda
.DEFAULT_GOAL := ayuda

# ---------------------------------------------------------------------------- #
ayuda:
	@echo "Flujo:  datos -> sinteticos -> estimar -> reporte"
	@echo ""
	@echo "  make setup                     instalar dependencias"
	@echo "  make all                       todo el flujo, todos los datasets"
	@echo "  make DATASET=lc_centro all     todo el flujo, un dataset"
	@echo "  make prueba                    corrida rapida de verificacion"
	@echo "  make limpiar                   borrar intermedios (data/processed, data/folds)"
	@echo ""
	@echo "Datasets: $(DATASETS)"
	@echo "Metodos:  $(METODOS)"

all: reporte

# ---------------------------------------------------------------------------- #
setup:
	$(PY) -m pip install -r requirements.txt
	$(RS) -e "if (!requireNamespace('renv', quietly=TRUE)) install.packages('renv'); renv::restore()"

verificar:
	@$(PY) python/scripts/00_verificar.py

# --- Paso 1: datos normalizados + folds -------------------------------------- #
# El centinela depende de config.yaml: si cambias el config, se rehace todo.
data/processed/MANIFEST.json: config.yaml python/scripts/01_folds.py
	$(PY) python/scripts/01_folds.py

datos: data/processed/MANIFEST.json

# --- Paso 2: conjuntos de entrenamiento por particion ------------------------ #
sinteticos: datos
	@for ds in $(OBJETIVOS); do \
		echo "=== sinteticos: $$ds ==="; \
		$(PY) python/scripts/02_sinteticos.py --dataset $$ds || exit 1; \
	done

# --- Paso 3: estimacion en R con Apollo -------------------------------------- #
estimar: sinteticos
	@for ds in $(OBJETIVOS); do \
		echo "=== estimar: $$ds ==="; \
		$(RS) R/03_estimar.R --dataset $$ds || exit 1; \
	done

# --- Paso 4: tablas finales -------------------------------------------------- #
reporte: estimar
	@for ds in $(OBJETIVOS); do \
		echo "=== reporte: $$ds ==="; \
		$(PY) python/scripts/04_reporte.py --dataset $$ds || exit 1; \
	done

# ---------------------------------------------------------------------------- #
#  Corrida rapida: 1 dataset, 2 metodos, 1 repeticion. Sirve para comprobar
#  que el flujo completo funciona antes de lanzar las 200 estimaciones.
# ---------------------------------------------------------------------------- #
prueba:
	$(PY) python/scripts/01_folds.py --dataset sm_centro
	$(PY) python/scripts/02_sinteticos.py --dataset sm_centro --metodo original
	$(PY) python/scripts/02_sinteticos.py --dataset sm_centro --metodo ROS
	$(RS) R/03_estimar.R --dataset sm_centro --metodo original --n-repeats 1
	$(RS) R/03_estimar.R --dataset sm_centro --metodo ROS --n-repeats 1
	$(PY) python/scripts/04_reporte.py --dataset sm_centro
	@echo ""
	@echo "Flujo verificado de punta a punta."

# ---------------------------------------------------------------------------- #
limpiar:
	rm -rf data/processed/* data/folds/* logs/apollo/*
	@echo "Intermedios borrados. Los datos crudos y los resultados siguen ahí."

limpiar-todo: limpiar
	rm -rf results/*.csv logs/*
	@echo "Resultados borrados también."
