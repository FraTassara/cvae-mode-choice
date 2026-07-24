# ============================================================================ #
#  R/config.R — espejo de python/cvae/config_repo.py
#
#  Lee el MISMO config.yaml. Ningún script de R contiene rutas ni semillas.
#
#  Nota: se usa here::here() en vez de
#      setwd(dirname(getActiveDocumentContext()$path))
#  porque esa forma requiere RStudio interactivo y hace imposible correr los
#  scripts con Rscript, que es lo que necesita `make`.
# ============================================================================ #

suppressPackageStartupMessages({
  library(here)
  library(yaml)
})

cargar_config <- function(path = here::here("config.yaml")) {
  cfg <- yaml::read_yaml(path)
  cfg$.path <- path
  cfg$.hash <- substr(
    digest::digest(readLines(path, warn = FALSE), algo = "sha256"), 1, 12
  )
  cfg
}

ruta <- function(cfg, clave, ...) {
  here::here(cfg$proyecto$rutas[[clave]], ...)
}

ruta_original <- function(cfg, ds) {
  ruta(cfg, "processed", sprintf("%s_original.csv", ds))
}

ruta_folds <- function(cfg, ds) {
  ruta(cfg, "folds", sprintf("%s_folds.csv", ds))
}

ruta_train <- function(cfg, ds, metodo) {
  ruta(cfg, "processed", sprintf("%s_%s_train.csv.gz", ds, metodo))
}

ruta_estimaciones <- function(cfg, ds) {
  ruta(cfg, "results", sprintf("estimaciones_%s.csv", ds))
}

#' Aborta si los datos intermedios no existen o son de otra versión del config.
verificar_datos <- function(cfg, ds) {
  manifest <- ruta(cfg, "processed", "MANIFEST.json")
  if (!file.exists(manifest)) {
    stop("Falta ", manifest, ". Corre `make datos` antes de estimar.",
         call. = FALSE)
  }
  for (f in c(ruta_original(cfg, ds), ruta_folds(cfg, ds))) {
    if (!file.exists(f)) stop("Falta ", f, call. = FALSE)
  }
  invisible(TRUE)
}
