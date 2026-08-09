#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------- #
#  estimar_caw.R  —  estima el MNL sobre los datos del CVAE con KL por clase
#                    (método CVAE_caw), de forma AISLADA.
#
#  Reutiliza tus funciones de R/ (estimar_mnl, predecir, calcular_metricas...).
#  Lee de experimental_results/<ds>_CVAE_caw_train.csv.gz (lo genera
#  experimental/exp_cv_caw.py) y ESCRIBE en experimental_results/, nunca en
#  results/. No toca tus estimaciones previas.
#
#  Uso:
#     Rscript experimental/estimar_caw.R --dataset sm_centro
# ---------------------------------------------------------------------------- #

suppressMessages({
  library(optparse)
  library(here)
})

# Reutilizamos la maquinaria del pipeline (rutas, spec, estimación, métricas).
source(here("R", "config.R"))
source(here("R", "apollo_modelo.R"))
source(here("R", "metricas.R"))

op <- OptionParser()
op <- add_option(op, "--dataset", type = "character", default = NULL)
op <- add_option(op, "--verbose", action = "store_true", default = FALSE)
args <- parse_args(op)
if (is.null(args$dataset)) stop("Falta --dataset", call. = FALSE)

cfg    <- cargar_config()
ds     <- args$dataset
metodo <- "CVAE_caw"

d          <- cfg$datasets[[ds]]
spec       <- d$modelo
choice_col <- d$choice_col

exp_dir <- here("experimental_results")
archivo_train <- file.path(exp_dir, paste0(ds, "_", metodo, "_train.csv.gz"))
if (!file.exists(archivo_train)) {
  stop("Falta ", basename(archivo_train),
       ". Corre antes: python experimental/exp_cv_caw.py --dataset ", ds,
       call. = FALSE)
}

original <- read.csv(ruta_original(cfg, ds))
folds    <- read.csv(ruta_folds(cfg, ds))
train_todos <- read.csv(gzfile(archivo_train))

dir_apollo <- file.path(exp_dir, "apollo_logs")
dir.create(dir_apollo, recursive = TRUE, showWarnings = FALSE)

particiones <- unique(folds[folds$split == "train", c("repeticion", "fold")])
particiones <- particiones[order(particiones$repeticion, particiones$fold), ]

message("[", ds, " / ", metodo, "] ", nrow(particiones),
        " particiones (aislado en experimental_results/)")

resultados <- list()
t_inicio <- Sys.time()

for (i in seq_len(nrow(particiones))) {
  rep_i  <- particiones$repeticion[i]
  fold_i <- particiones$fold[i]

  tr <- train_todos[train_todos$repeticion == rep_i &
                      train_todos$fold == fold_i, ]
  tr <- tr[, !(names(tr) %in% c("repeticion", "fold"))]
  tr$row_id <- seq_len(nrow(tr))

  ids_val <- folds$row_id[folds$repeticion == rep_i &
                            folds$fold == fold_i &
                            folds$split == "valid"]
  va <- original[original$row_id %in% ids_val, ]

  etiqueta <- sprintf("%s_r%02d_f%d", metodo, rep_i, fold_i)

  fila <- tryCatch({
    ajuste <- estimar_mnl(tr, spec, choice_col,
                          nombre_modelo = etiqueta,
                          directorio_salida = dir_apollo,
                          silencioso = !args$verbose)
    P  <- predecir(ajuste, va, spec, choice_col)
    m  <- calcular_metricas(P, va[[choice_col]])
    co <- extraer_coeficientes(ajuste$modelo)
    c(list(dataset = ds, metodo = metodo,
           repeticion = rep_i, fold = fold_i,
           n_train = nrow(tr),
           n_sinteticos = sum(tr$is_synthetic, na.rm = TRUE)),
      m, co, list(vot = calcular_vot(co)))
  }, error = function(e) {
    message("  ", etiqueta, ": ERROR — ", conditionMessage(e)); NULL
  })

  if (!is.null(fila)) resultados[[length(resultados) + 1]] <- fila
  if (i %% 5 == 0 || i == nrow(particiones)) {
    message("  ", i, "/", nrow(particiones), "  (",
            round(difftime(Sys.time(), t_inicio, units = "mins"), 1), " min)")
  }
}

if (length(resultados) == 0) stop("No se estimó ningún modelo.", call. = FALSE)

todas <- unique(unlist(lapply(resultados, names)))
tabla <- do.call(rbind, lapply(resultados, function(r) {
  r[setdiff(todas, names(r))] <- NA
  as.data.frame(r[todas], stringsAsFactors = FALSE)
}))

destino <- file.path(exp_dir, paste0("estimaciones_", ds, "_", metodo, ".csv"))
write.csv(tabla, destino, row.names = FALSE)
message("\n", nrow(tabla), " estimaciones -> experimental_results/",
        basename(destino))
