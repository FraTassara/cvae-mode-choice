#!/usr/bin/env Rscript
# ============================================================================ #
#  R/03_estimar.R — Paso 3: estimar el MNL en cada partición.
#
#  Un solo script parametrizado. Reemplaza los 13 archivos .R que tenían
#  hasta 7 líneas `database = read.csv(...)` sin comentar, donde ganaba la
#  última y era imposible saber con qué datos se produjo cada número.
#
#  Uso:
#      Rscript R/03_estimar.R --dataset lc_centro
#      Rscript R/03_estimar.R --dataset lc_centro --metodo CVAE
#      Rscript R/03_estimar.R --dataset lc_centro --n-repeats 2   # prueba rápida
# ============================================================================ #

suppressPackageStartupMessages({
  library(optparse)
  library(here)
})

source(here::here("R", "config.R"))
source(here::here("R", "apollo_modelo.R"))
source(here::here("R", "metricas.R"))

# ---------------------------------------------------------------------------- #
opciones <- list(
  make_option("--dataset", type = "character", help = "clave en config.yaml"),
  make_option("--metodo",  type = "character", default = NULL,
              help = "por defecto, todos los de config.yaml"),
  make_option("--n-repeats", type = "integer", default = NULL,
              help = "limitar repeticiones (para pruebas)"),
  make_option("--verbose", action = "store_true", default = FALSE)
)
args <- parse_args(OptionParser(option_list = opciones))
if (is.null(args$dataset)) stop("Falta --dataset", call. = FALSE)

cfg <- cargar_config()
verificar_datos(cfg, args$dataset)

d          <- cfg$datasets[[args$dataset]]
spec       <- d$modelo
choice_col <- d$choice_col
metodos    <- if (is.null(args$metodo)) unlist(cfg$metodos) else args$metodo

original <- read.csv(ruta_original(cfg, args$dataset))
folds    <- read.csv(ruta_folds(cfg, args$dataset))

dir_apollo <- ruta(cfg, "logs", "apollo")
dir.create(dir_apollo, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------- #
resultados <- list()
t_inicio <- Sys.time()

for (metodo in metodos) {

  archivo_train <- ruta_train(cfg, args$dataset, metodo)
  if (!file.exists(archivo_train)) {
    message("SALTADO ", metodo, ": falta ", basename(archivo_train),
            " (corre `make sinteticos`)")
    next
  }
  train_todos <- read.csv(gzfile(archivo_train))

  particiones <- unique(folds[folds$split == "train", c("repeticion", "fold")])
  particiones <- particiones[order(particiones$repeticion, particiones$fold), ]
  if (!is.null(args$`n-repeats`)) {
    particiones <- particiones[particiones$repeticion < args$`n-repeats`, ]
  }

  message("\n[", args$dataset, " / ", metodo, "] ",
          nrow(particiones), " particiones")

  for (i in seq_len(nrow(particiones))) {
    rep_i  <- particiones$repeticion[i]
    fold_i <- particiones$fold[i]

    # --- datos de entrenamiento de ESTA partición (con sus sintéticos) ------
    tr <- train_todos[train_todos$repeticion == rep_i &
                        train_todos$fold == fold_i, ]
    tr <- tr[, !(names(tr) %in% c("repeticion", "fold"))]
    # Apollo necesita un indivID único; las filas sintéticas traen row_id = -1.
    tr$row_id <- seq_len(nrow(tr))

    # --- validación: SOLO filas reales, nunca sintéticas --------------------
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

      P <- predecir(ajuste, va, spec, choice_col)
      m <- calcular_metricas(P, va[[choice_col]])
      co <- extraer_coeficientes(ajuste$modelo)

      c(list(dataset = args$dataset, metodo = metodo,
             repeticion = rep_i, fold = fold_i,
             n_train = nrow(tr),
             n_sinteticos = sum(tr$is_synthetic, na.rm = TRUE)),
        m, co,
        list(vot = calcular_vot(co)))
    }, error = function(e) {
      message("  ", etiqueta, ": ERROR — ", conditionMessage(e))
      NULL
    })

    if (!is.null(fila)) resultados[[length(resultados) + 1]] <- fila

    if (i %% 5 == 0 || i == nrow(particiones)) {
      message("  ", i, "/", nrow(particiones), "  (",
              round(difftime(Sys.time(), t_inicio, units = "mins"), 1), " min)")
    }
  }
}

# ---------------------------------------------------------------------------- #
if (length(resultados) == 0) stop("No se estimó ningún modelo.", call. = FALSE)

# rbind tolerante: distintos métodos pueden tener distintos coeficientes
todas_las_columnas <- unique(unlist(lapply(resultados, names)))
tabla <- do.call(rbind, lapply(resultados, function(r) {
  faltan <- setdiff(todas_las_columnas, names(r))
  r[faltan] <- NA
  as.data.frame(r[todas_las_columnas], stringsAsFactors = FALSE)
}))

destino <- ruta_estimaciones(cfg, args$dataset)

# Acumular en vez de sobrescribir: estimar un método nuevo (class_weights,
# CVAE_conf90, ...) no debe borrar los ya calculados. Se reemplazan solo las
# filas de los métodos que se acaban de estimar.
if (file.exists(destino)) {
  previo <- read.csv(destino, stringsAsFactors = FALSE)
  previo <- previo[!(previo$metodo %in% unique(tabla$metodo)), , drop = FALSE]
  if (nrow(previo) > 0) {
    faltan_en_previo <- setdiff(names(tabla), names(previo))
    for (col in faltan_en_previo) previo[[col]] <- NA
    faltan_en_tabla <- setdiff(names(previo), names(tabla))
    for (col in faltan_en_tabla) tabla[[col]] <- NA
    tabla <- rbind(previo[, names(tabla), drop = FALSE], tabla)
  }
  message("Acumulado con resultados previos (",
          length(unique(tabla$metodo)), " métodos en total).")
}

write.csv(tabla, destino, row.names = FALSE)
message("\n", nrow(tabla), " estimaciones -> ", destino)

# Constancia de versiones, para replicabilidad.
writeLines(
  capture.output(sessionInfo()),
  ruta(cfg, "logs", sprintf("sessionInfo_%s.txt", args$dataset))
)