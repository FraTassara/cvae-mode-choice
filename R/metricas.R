# ============================================================================ #
#  R/metricas.R
#
#  Una sola función, con la orientación de la matriz de confusión FIJA.
#  El código original tenía dos versiones que construían la tabla al revés
#  (`table(pred, actual)` vs `table(actual, pred)`), con lo que precision y
#  recall quedaban intercambiados entre bloques, y una tercera que calculaba
#  `recall[i] = sensitivity[i] / sum(cm[, i])`, que no es ninguna métrica
#  conocida (sensitivity ya ES el recall).
# ============================================================================ #

#' Métricas de un conjunto de validación.
#'
#' @param P matriz n x J de probabilidades, colnames = ids de alternativa
#' @param y vector de alternativas elegidas (mismos ids)
calcular_metricas <- function(P, y) {

  alt_ids <- as.integer(colnames(P))
  stopifnot(nrow(P) == length(y))

  # --- probabilidad de la alternativa efectivamente elegida ----------------
  idx <- match(as.integer(y), alt_ids)
  p_obs <- P[cbind(seq_len(nrow(P)), idx)]
  p_obs <- pmax(p_obs, 1e-12)

  # --- alternativa predicha (argmax) ---------------------------------------
  pred <- alt_ids[max.col(P, ties.method = "first")]

  # --- matriz de confusión: SIEMPRE (real en filas, predicho en columnas) ---
  niveles <- sort(alt_ids)
  cm <- table(
    real     = factor(y,    levels = niveles),
    predicho = factor(pred, levels = niveles)
  )

  diagonal <- diag(cm)
  por_real     <- rowSums(cm)   # cuántas hay de cada clase real
  por_predicho <- colSums(cm)   # cuántas se predijeron de cada clase

  recall    <- ifelse(por_real > 0,     diagonal / por_real, NA_real_)
  precision <- ifelse(por_predicho > 0, diagonal / por_predicho, NA_real_)
  f1 <- ifelse(is.na(precision) | is.na(recall) | (precision + recall) == 0,
               0, 2 * precision * recall / (precision + recall))

  # --- norma L1 de market share --------------------------------------------
  share_real  <- as.numeric(por_real / sum(por_real))
  share_pred  <- colMeans(P)
  l1 <- sum(abs(share_real - share_pred))

  salida <- list(
    accuracy             = sum(diagonal) / sum(cm),
    f1_macro             = mean(f1, na.rm = TRUE),
    loglik_holdout       = sum(log(p_obs)),
    loglik_media         = mean(log(p_obs)),
    prob_media_observada = mean(p_obs),
    l1_market_share      = l1,
    n_valid              = nrow(P)
  )

  # métricas por clase, con el id de alternativa en el nombre
  for (i in seq_along(niveles)) {
    a <- niveles[i]
    salida[[paste0("precision_", a)]] <- precision[i]
    salida[[paste0("recall_", a)]]    <- recall[i]
    salida[[paste0("f1_", a)]]        <- f1[i]
  }

  salida
}

#' Extrae coeficientes y errores estándar POR NOMBRE.
#'
#' El código original hacía `b <- apollo_modelOutput(m)[, 1]` y luego indexaba
#' por posición (`b[9] * CTOT1_w`). Si cambia el orden o el número de
#' parámetros, las utilidades quedan mal sin ningún error.
extraer_coeficientes <- function(modelo) {
  est <- modelo$estimate
  se  <- tryCatch(modelo$robse, error = function(e) modelo$se)
  if (is.null(se)) se <- modelo$se

  salida <- list()
  for (nm in names(est)) {
    salida[[paste0("beta_", nm)]] <- unname(est[[nm]])
    salida[[paste0("se_", nm)]]   <- unname(se[[nm]])
  }
  salida$loglik_train <- modelo$maximum
  salida$loglik_nula  <- modelo$LL0
  salida$rho2         <- 1 - modelo$maximum / modelo$LL0
  salida$convergio    <- isTRUE(modelo$successfulEstimation)
  salida
}

#' Valor del tiempo, por nombre de coeficiente.
calcular_vot <- function(coefs, beta_tiempo = "b_ivt", beta_costo = "b_tcw") {
  bt <- coefs[[paste0("beta_", beta_tiempo)]]
  bc <- coefs[[paste0("beta_", beta_costo)]]
  if (is.null(bt) || is.null(bc) || is.na(bc) || bc == 0) return(NA_real_)
  bt / bc
}
