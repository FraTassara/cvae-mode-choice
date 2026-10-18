#!/usr/bin/env Rscript
# ============================================================================ #
#  R/08_efectos_marginales.R — §2.3(c) del plan de revisores.
#
#  Sugerencia explícita del editor: variar cada covariable de a una, mantener
#  las demás fijas, y observar cómo cambian las probabilidades predichas.
#
#  Compara el efecto marginal del modelo estimado con datos ORIGINALES contra
#  el de los modelos estimados con cada método de balanceo. Si el CVAE altera
#  la sensibilidad del modelo al tiempo o al costo, aquí se ve directamente,
#  sin depender del signo del coeficiente.
#
#  N9 (corrigendum): antes se estimaba UN modelo por método, en la partición
#  (0, 0), y la Tabla 9 dependía de esa partición (en SM-Centro el CVAE tenía
#  ahí b_ivt > 0 aunque en media sobre las 50 no está invertido). Ahora se
#  estima en TODAS las particiones de entrenamiento, con la misma regla que
#  04_reporte.py: entran todas, se informa `convergio` y solo se descartan los
#  ajustes sin coeficientes finitos.
#
#  Salidas:
#      efectos_marginales_{ds}.csv          una fila por partición x método x
#                                           covariable x alternativa
#      efectos_marginales_resumen_{ds}.csv  media, SD, n y % de particiones
#                                           con elasticidad positiva
#
#  Uso:
#      Rscript R/08_efectos_marginales.R --dataset sm_centro
#      Rscript R/08_efectos_marginales.R --dataset sm_centro --delta 0.10
#      Rscript R/08_efectos_marginales.R --dataset sm_centro --n-repeats 1   # prueba
# ============================================================================ #

suppressPackageStartupMessages({
  library(optparse)
  library(here)
})

source(here::here("R", "config.R"))
source(here::here("R", "apollo_modelo.R"))

opciones <- list(
  make_option("--dataset", type = "character"),
  make_option("--delta", type = "double", default = 0.01,
              help = "perturbación relativa de la covariable (0.01 = +1%)"),
  make_option("--n-repeats", type = "integer", default = NULL,
              help = "limitar repeticiones (para pruebas)")
)
args <- parse_args(OptionParser(option_list = opciones))
if (is.null(args$dataset)) stop("Falta --dataset", call. = FALSE)

cfg <- cargar_config()
verificar_datos(cfg, args$dataset)

d          <- cfg$datasets[[args$dataset]]
spec       <- d$modelo
choice_col <- d$choice_col

original <- read.csv(ruta_original(cfg, args$dataset))
folds    <- read.csv(ruta_folds(cfg, args$dataset))
particiones <- unique(folds[folds$split == "train", c("repeticion", "fold")])
particiones <- particiones[order(particiones$repeticion, particiones$fold), ]
if (!is.null(args$`n-repeats`)) {
  particiones <- particiones[particiones$repeticion < args$`n-repeats`, ]
}
dir_apollo <- ruta(cfg, "logs", "apollo")
dir.create(dir_apollo, recursive = TRUE, showWarnings = FALSE)

# Covariables continuas que aparecen en alguna utilidad.
covariables <- unique(unlist(lapply(spec$alternativas, function(a) unlist(a$atributos))))
covariables <- covariables[vapply(covariables, function(v) {
  v %in% names(original) && is.numeric(original[[v]]) &&
    length(unique(original[[v]])) > 2      # excluye binarias
}, logical(1))]

message("Covariables a perturbar: ", paste(covariables, collapse = ", "))

ids <- vapply(spec$alternativas, function(a) a$nombre, character(1))
alt_ids <- vapply(spec$alternativas, function(a) as.character(a$id), character(1))

resultados <- list()
descartadas <- list()

for (metodo in unlist(cfg$metodos)) {

  archivo <- ruta_train(cfg, args$dataset, metodo)
  if (!file.exists(archivo)) {
    message("SALTADO ", metodo, ": falta ", basename(archivo))
    next
  }

  train_todos <- read.csv(gzfile(archivo))
  message("\n[", metodo, "] ", nrow(particiones), " particiones")
  descartadas[[metodo]] <- 0L

  for (i in seq_len(nrow(particiones))) {
    rep_i  <- particiones$repeticion[i]
    fold_i <- particiones$fold[i]

    tr <- train_todos[train_todos$repeticion == rep_i &
                        train_todos$fold == fold_i, ]
    tr <- tr[, !(names(tr) %in% c("repeticion", "fold"))]
    tr$row_id <- seq_len(nrow(tr))

    ajuste <- tryCatch(
      estimar_mnl(tr, spec, choice_col,
                  nombre_modelo = sprintf("efmarg_%s_r%02d_f%d", metodo, rep_i, fold_i),
                  directorio_salida = dir_apollo, silencioso = TRUE),
      error = function(e) { message("  error: ", conditionMessage(e)); NULL })

    # Misma regla que python/cvae/convergencia.py: se descarta solo el ajuste
    # sin coeficientes finitos; `convergio` se registra, no filtra.
    if (is.null(ajuste) || !all(is.finite(ajuste$modelo$estimate))) {
      descartadas[[metodo]] <- descartadas[[metodo]] + 1L
      next
    }
    convergio <- isTRUE(ajuste$modelo$successfulEstimation)

    # --- Cuotas base: predicción sobre los datos ORIGINALES sin perturbar -----
    P0 <- predecir(ajuste, original, spec, choice_col)
    base <- colMeans(P0)

    # --- Perturbar cada covariable de a una ----------------------------------
    for (v in covariables) {
      datos_mod <- original
      incremento <- args$delta * mean(datos_mod[[v]], na.rm = TRUE)
      datos_mod[[v]] <- datos_mod[[v]] + incremento

      P1 <- tryCatch(predecir(ajuste, datos_mod, spec, choice_col),
                     error = function(e) NULL)
      if (is.null(P1)) next
      nuevo <- colMeans(P1)

      for (j in seq_along(alt_ids)) {
        # Elasticidad arco: cambio % en la probabilidad por cambio % en la
        # covariable. Es adimensional, así que se puede comparar entre
        # variables con unidades distintas (minutos vs. pesos).
        elast <- ((nuevo[j] - base[j]) / base[j]) / args$delta
        resultados[[length(resultados) + 1]] <- data.frame(
          dataset     = args$dataset,
          metodo      = metodo,
          repeticion  = rep_i,
          fold        = fold_i,
          convergio   = convergio,
          covariable  = v,
          alternativa = alt_ids[j],
          prob_base   = base[j],
          prob_perturbada = nuevo[j],
          cambio_abs  = nuevo[j] - base[j],
          elasticidad = elast,
          stringsAsFactors = FALSE
        )
      }
    }

    if (i %% 10 == 0 || i == nrow(particiones)) {
      message("  ", i, "/", nrow(particiones))
    }
  }
}

if (length(resultados) == 0) stop("No se pudo estimar ningún modelo.", call. = FALSE)

tabla <- do.call(rbind, resultados)
destino <- ruta(cfg, "results", sprintf("efectos_marginales_%s.csv", args$dataset))
dir.create(dirname(destino), recursive = TRUE, showWarnings = FALSE)
write.csv(tabla, destino, row.names = FALSE)

message("\n", nrow(tabla), " filas -> ", destino)

# --- Resumen sobre las particiones ------------------------------------------
claves <- c("metodo", "covariable", "alternativa")
grupos <- split(tabla, tabla[, claves], drop = TRUE)
resumen <- do.call(rbind, lapply(grupos, function(g) {
  data.frame(
    dataset            = args$dataset,
    metodo             = g$metodo[1],
    covariable         = g$covariable[1],
    alternativa        = g$alternativa[1],
    elasticidad_media  = mean(g$elasticidad),
    elasticidad_sd     = sd(g$elasticidad),
    pct_positiva       = mean(g$elasticidad > 0),
    n_particiones      = nrow(g),
    n_no_convergidas   = sum(!g$convergio),
    n_descartadas      = descartadas[[g$metodo[1]]],
    stringsAsFactors   = FALSE
  )
}))
resumen <- resumen[order(resumen$covariable, resumen$alternativa, resumen$metodo), ]
destino_res <- ruta(cfg, "results",
                    sprintf("efectos_marginales_resumen_%s.csv", args$dataset))
write.csv(resumen, destino_res, row.names = FALSE)
message(nrow(resumen), " filas -> ", destino_res)

message("\n--- Elasticidad media sobre particiones (cambio % en probabilidad por +1%) ---")
ancho <- reshape(resumen[, c(claves, "elasticidad_media")],
                 idvar = c("covariable", "alternativa"),
                 timevar = "metodo", direction = "wide")
names(ancho) <- gsub("elasticidad_media.", "", names(ancho), fixed = TRUE)
print(format(ancho, digits = 3), row.names = FALSE)

message("\nComparar la columna 'original' con las demás: si un método cambia")
message("el SIGNO o la MAGNITUD de la elasticidad, distorsionó la respuesta")
message("del modelo a esa variable, no solo el coeficiente.")
