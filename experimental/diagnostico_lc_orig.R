# ============================================================================ #
#  experimental/diagnostico_lc_orig.R — N10 del corrigendum (diagnóstico único).
#
#  ¿De qué estimación sale la columna ORIG. de LC-Centro en la Tabla 11
#  publicada? Estima variantes de la especificación EN MEMORIA, sobre la muestra
#  completa, sin tocar config.yaml (la spec de LC no se modifica). Las columnas
#  de las variantes (ILM, JEFH, modo 3) se nombran aquí a propósito: es un
#  diagnóstico, no parte del pipeline.
#
#  Uso:  Rscript experimental/diagnostico_lc_orig.R
#  Salida: docs/diagnostico_lc_orig.csv
# ============================================================================ #
suppressPackageStartupMessages(library(here))
source(here::here("R", "config.R"))
source(here::here("R", "apollo_modelo.R"))
source(here::here("R", "metricas.R"))

cfg  <- cargar_config()
d    <- cfg$datasets$lc_centro
base <- d$modelo
crudo <- read.csv(here::here(cfg$proyecto$rutas$raw, d$archivo))
orig  <- read.csv(ruta_original(cfg, "lc_centro"))

con_attr <- function(spec, alt_id, nuevos) {
  for (i in seq_along(spec$alternativas)) {
    if (spec$alternativas[[i]]$id == alt_id)
      spec$alternativas[[i]]$atributos <- c(spec$alternativas[[i]]$atributos, nuevos)
  }
  spec
}
mi <- function(spec) {
  spec <- con_attr(spec, 1, list(b_mi_cd = "ILM"))
  spec <- con_attr(spec, 2, list(b_mi_cp = "ILM"))
  spec <- con_attr(spec, 5, list(b_mi_bus = "ILM"))
  spec$escalas <- list(ILM = 1e-4)
  spec
}
sin_avail <- function(spec) {
  for (i in seq_along(spec$alternativas)) spec$alternativas[[i]]$avail <- NULL
  spec
}
con_modo3 <- function(spec) {
  spec$alternativas[[length(spec$alternativas) + 1]] <- list(
    id = 3, nombre = "modo3", avail = "DISP3",
    atributos = list(b_tcw = "CTOT3_w", b_ivt = "TVIA3", b_wkt = "TCAM3", b_wt = "TESP3"))
  spec
}

variantes <- list(
  V0_pipeline        = list(spec = base, datos = orig),
  V1_mas_MI          = list(spec = mi(base), datos = orig),
  V2_MI_y_HH_cp      = list(spec = con_attr(mi(base), 2, list(b_hh_cp = "JEFH")), datos = orig),
  V3_sin_avail       = list(spec = sin_avail(base), datos = orig),
  V4_N444_con_modo3  = list(spec = con_modo3(base), datos = crudo),
  V5_N444_MI_modo3   = list(spec = con_modo3(mi(base)), datos = crudo)
)

dir_apollo <- file.path(tempdir(), "apollo_diag"); dir.create(dir_apollo, showWarnings = FALSE)
filas <- list()
for (nm in names(variantes)) {
  v <- variantes[[nm]]
  datos <- v$datos; datos$row_id <- seq_len(nrow(datos))
  aj <- tryCatch(estimar_mnl(datos, v$spec, d$choice_col, nombre_modelo = nm,
                             directorio_salida = dir_apollo, silencioso = TRUE),
                 error = function(e) { message(nm, ": ", conditionMessage(e)); NULL })
  if (is.null(aj)) next
  est <- aj$modelo$estimate; se <- aj$modelo$se
  for (p in names(est)) {
    filas[[length(filas) + 1]] <- data.frame(variante = nm, n = nrow(datos),
      parametro = p, beta = unname(est[p]), se = unname(se[p]),
      ll = aj$modelo$maximum, convergio = isTRUE(aj$modelo$successfulEstimation))
  }
}
res <- do.call(rbind, filas)
write.csv(res, here::here("docs", "diagnostico_lc_orig.csv"), row.names = FALSE)
cat("ok\n")
