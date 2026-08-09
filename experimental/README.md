# experimental/ — CVAE mejorado (módulo acoplable)

Componente **aparte** que prueba dos mejoras del CVAE sin tocar el pipeline
validado. Importa de `cvae` todo lo reutilizable (datos, folds, elipses,
diagnósticos, generación) y solo reimplementa lo que cambia.

## Las dos mejoras

1. **KL annealing por clase** (`model_class_aware.py`). Relaja el peso KL de las
   clases minoritarias para evitar que su latente colapse al prior. Peso por
   clase = `kl_min + (1-kl_min)·(n_c/n_max)`: la mayoritaria mantiene KL=1, la
   minoritaria baja hasta `kl_min`. Con `kl_min=1` se recupera el CVAE base.
   (Base: Wang et al. 2025, CAMIL — class-aware KL weight.)

2. **Espacio latente de dimensión alta** con visualización por PCA
   (`plots_pca.py`). El muestreo gaussiano por elipse ya funciona en cualquier
   `n_z`; lo único que hace falta para `n_z > 2` es proyectar a 2D. Se usa PCA
   (lineal) para que la elipse se proyecte como elipse y la figura sea
   comparable con la del base (`n_z = 2`).

## Uso

Desde la raíz del repo, con el `.venv` activado:

```bash
python experimental/exp_comparar.py --dataset sm_centro
python experimental/exp_comparar.py --dataset sm_centro --n-z 8 --kl-min 0.3
```

Compara el CVAE base (n_z=2, KL global) contra el mejorado sobre las mismas
observaciones. Produce:

- **Tabla de JSD de la clase minoritaria** — la métrica que prueba si las
  mejoras generan mejor la clase rara. Menor es mejor; el Δ positivo significa
  que el modelo mejorado gana.
- **Chequeo de Mahalanobis** — confirma que el muestreo llena bien el elipsoide
  en `n_z` dimensiones (la fracción empírica dentro de la elipse debe acercarse
  al nivel de confianza teórico).
- **Dos figuras del espacio latente** — base en 2D directo, mejorado en PCA 2D
  con el % de varianza explicada anotado.

## Qué NO hace

- No modifica ningún archivo de `python/cvae/` ni del pipeline.
- No corre la validación cruzada completa ni estima el MNL: es un experimento de
  concepto sobre un dataset, no una evaluación para el paper.
- Requiere `n_z = 2` en el modelo base (para su figura directa); el mejorado
  admite cualquier `n_z`.

## Si funciona

Si el JSD de la minoritaria baja de forma consistente, el siguiente paso sería
integrar las mejoras al pipeline principal (nuevo método en `02_sinteticos.py`)
y correr la evaluación completa con validación cruzada. Eso ya sí tocaría el
pipeline, así que se haría con la misma cautela que el resto del proyecto.
