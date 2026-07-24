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
#  Uso:
#      Rscript R/08_efectos_marginales.R --dataset sm_centro
#      Rscript R/08_efectos_marginales.R --dataset sm_centro --delta 0.10
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
  make_option("--repeticion", type = "integer", default = 0,
              help = "qué repetición usar para estimar (por defecto la 0)"),
  make_option("--fold", type = "integer", default = 0)
)
args <- parse_args(OptionParser(option_list = opciones))
if (is.null(args$dataset)) stop("Falta --dataset", call. = FALSE)

cfg <- cargar_config()
verificar_datos(cfg, args$dataset)

d          <- cfg$datasets[[args$dataset]]
spec       <- d$modelo
choice_col <- d$choice_col

original <- read.csv(ruta_original(cfg, args$dataset))
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

for (metodo in unlist(cfg$metodos)) {

  archivo <- ruta_train(cfg, args$dataset, metodo)
  if (!file.exists(archivo)) {
    message("SALTADO ", metodo, ": falta ", basename(archivo))
    next
  }

  train_todos <- read.csv(gzfile(archivo))
  tr <- train_todos[train_todos$repeticion == args$repeticion &
                      train_todos$fold == args$fold, ]
  tr <- tr[, !(names(tr) %in% c("repeticion", "fold"))]
  tr$row_id <- seq_len(nrow(tr))

  message("\n[", metodo, "] estimando sobre ", nrow(tr), " filas...")
  ajuste <- tryCatch(
    estimar_mnl(tr, spec, choice_col,
                nombre_modelo = paste0("efmarg_", metodo),
                directorio_salida = dir_apollo, silencioso = TRUE),
    error = function(e) { message("  error: ", conditionMessage(e)); NULL })
  if (is.null(ajuste)) next

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
}

if (length(resultados) == 0) stop("No se pudo estimar ningún modelo.", call. = FALSE)

tabla <- do.call(rbind, resultados)
destino <- ruta(cfg, "results", sprintf("efectos_marginales_%s.csv", args$dataset))
write.csv(tabla, destino, row.names = FALSE)

message("\n", nrow(tabla), " filas -> ", destino)

# Resumen: elasticidad propia (la de la covariable sobre su propia alternativa)
message("\n--- Elasticidades (cambio % en probabilidad por +1% en la covariable) ---")
resumen <- reshape(tabla[, c("metodo", "covariable", "alternativa", "elasticidad")],
                   idvar = c("covariable", "alternativa"),
                   timevar = "metodo", direction = "wide")
names(resumen) <- gsub("elasticidad.", "", names(resumen), fixed = TRUE)
print(format(resumen, digits = 3), row.names = FALSE)

message("\nComparar la columna 'original' con las demás: si un método cambia")
message("el SIGNO o la MAGNITUD de la elasticidad, distorsionó la respuesta")
message("del modelo a esa variable, no solo el coeficiente.")
