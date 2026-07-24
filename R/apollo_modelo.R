# ============================================================================ #
#  R/apollo_modelo.R — construye la especificación de Apollo desde config.yaml
#
#  Reemplaza los bloques `apollo_probabilities` copiados en 13 archivos .R.
#  La especificación existe UNA sola vez: la de estimación y la de predicción
#  son literalmente la misma función, así que no pueden divergir.
# ============================================================================ #

suppressPackageStartupMessages({
  library(apollo)
})

#' Vector de parámetros iniciales a partir de la config.
#'
#' @param spec cfg$datasets[[ds]]$modelo
construir_beta_inicial <- function(spec) {
  ascs <- setNames(
    rep(0, length(spec$alternativas)),
    vapply(spec$alternativas, function(a) paste0("asc_", a$nombre), "")
  )
  # Betas genéricos: si el mismo nombre aparece en varias alternativas, se
  # estima UNO solo (comportamiento de Apollo y del código original).
  nombres_beta <- unique(unlist(lapply(spec$alternativas, function(a) names(a$atributos))))
  betas <- setNames(rep(0, length(nombres_beta)), nombres_beta)
  c(ascs, betas)
}

#' Genera la función apollo_probabilities para un dataset.
#'
#' Se construye por metaprogramación a partir de la config, en vez de escribir
#' a mano `V[['car_driver']] = asc_car_driver + b_tcw * CTOT1_w + ...` en cada
#' archivo. Cambiar la especificación = editar config.yaml.
construir_probabilities <- function(spec, choice_col, usar_pesos = FALSE) {

  alternativas <- setNames(
    vapply(spec$alternativas, function(a) as.integer(a$id), integer(1)),
    vapply(spec$alternativas, function(a) a$nombre, character(1))
  )

  # Expresiones de utilidad como texto, evaluadas dentro del entorno de Apollo.
  utilidades <- lapply(spec$alternativas, function(a) {
    terminos <- c(paste0("asc_", a$nombre))
    for (b in names(a$atributos)) {
      terminos <- c(terminos, sprintf("%s * %s", b, a$atributos[[b]]))
    }
    paste(terminos, collapse = " + ")
  })
  names(utilidades) <- vapply(spec$alternativas, function(a) a$nombre, "")

  disponibilidad <- lapply(spec$alternativas, function(a) a$avail)
  names(disponibilidad) <- names(utilidades)

  function(apollo_beta, apollo_inputs, functionality = "estimate") {
    apollo_attach(apollo_beta, apollo_inputs)
    on.exit(apollo_detach(apollo_beta, apollo_inputs))

    P <- list()
    V <- list()
    for (nm_v in names(utilidades)) {
      V[[nm_v]] <- eval(parse(text = utilidades[[nm_v]]))
    }

    av <- list()
    for (nm_a in names(disponibilidad)) {
      col <- disponibilidad[[nm_a]]
      av[[nm_a]] <- if (is.null(col)) 1 else eval(parse(text = col))
    }

    mnl_settings <- list(
      alternatives = alternativas,
      avail        = av,
      choiceVar    = eval(parse(text = choice_col)),
      V            = V
    )

    P[["model"]] <- apollo_mnl(mnl_settings, functionality)
    # La ponderación debe aplicarse ANTES de apollo_prepareProb.
    if (usar_pesos) P <- apollo_weighting(P, apollo_inputs, functionality)
    P <- apollo_prepareProb(P, apollo_inputs, functionality)
    return(P)
  }
}

#' Estima el MNL sobre `datos` y devuelve el modelo de Apollo.
estimar_mnl <- function(datos, spec, choice_col, nombre_modelo,
                        directorio_salida, silencioso = TRUE) {

  apollo_initialise()

  apollo_control <- list(
    modelName       = nombre_modelo,
    modelDescr      = nombre_modelo,
    indivID         = "row_id",
    panelData       = FALSE,
    mixing          = FALSE,
    nCores          = 1,          # el paralelismo va a nivel de fold
    outputDirectory = directorio_salida,
    noValidation    = TRUE,
    noDiagnostics   = TRUE
  )

  # §2.9 — Si el bloque trae columna `peso` con valores no constantes, se
  # activa la verosimilitud ponderada de Apollo (baseline con class weights).
  usar_pesos <- "peso" %in% names(datos) &&
    length(unique(datos$peso)) > 1
  if (usar_pesos) apollo_control$weights <- "peso"

  apollo_beta  <- construir_beta_inicial(spec)
  apollo_fixed <- spec$asc_fija

  database <<- datos   # Apollo espera `database` en el entorno global

  apollo_inputs <- apollo_validateInputs(
    database      = datos,
    apollo_beta   = apollo_beta,
    apollo_fixed  = apollo_fixed,
    apollo_control = apollo_control
  )

  probs <- construir_probabilities(spec, choice_col, usar_pesos)

  estimate_settings <- list(
    writeIter    = FALSE,
    silent       = silencioso,
    hessianRoutine = "analytic"
  )

  modelo <- apollo_estimate(apollo_beta, apollo_fixed, probs,
                            apollo_inputs, estimate_settings)
  list(modelo = modelo, probs = probs, inputs = apollo_inputs,
       beta = apollo_beta, fixed = apollo_fixed, control = apollo_control)
}

#' Predice sobre un conjunto NUEVO reutilizando la misma especificación.
#'
#' Esto reemplaza el bucle manual del código original, que reconstruía el logit
#' a mano con índices posicionales (`b[1]`, `b[6]`, `b[9]`) y por tanto podía
#' desincronizarse de apollo_probabilities sin dar ningún error.
predecir <- function(ajuste, datos_nuevos, spec, choice_col) {

  database <<- datos_nuevos

  # La ponderación es un asunto de ESTIMACIÓN, no de predicción: el conjunto
  # de validación son filas reales sin ponderar y no trae columna `peso`.
  control_pred <- ajuste$control
  control_pred$weights <- NULL

  inputs_nuevos <- apollo_validateInputs(
    database       = datos_nuevos,
    apollo_beta    = ajuste$beta,
    apollo_fixed   = ajuste$fixed,
    apollo_control = control_pred
  )

  pred <- apollo_prediction(
    ajuste$modelo, construir_probabilities(spec, choice_col, usar_pesos = FALSE),
    inputs_nuevos,
    prediction_settings = list(silent = TRUE)
  )

  # apollo_prediction devuelve ID, Observation, una columna por alternativa
  # y 'chosen'. Nos quedamos con las probabilidades por alternativa.
  ids <- vapply(spec$alternativas, function(a) a$nombre, character(1))
  P <- as.matrix(pred[, ids, drop = FALSE])
  colnames(P) <- vapply(spec$alternativas, function(a) as.character(a$id), "")
  P
}