pacman::p_load(dplyr, projpred)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

conditional_effects_proj <- function(prjfit, 
                                     newdata_template, 
                                     predictor, 
                                     grid_length = 50, 
                                     ci_level = 0.95, 
                                     integrated = FALSE, 
                                     seed = 1234) {
  
  # prjfit: projected model (from project())
  # newdata_template: data.frame with all predictors; non-target predictors used as baseline
  # predictor: string, predictor to generate conditional effects for
  # grid_length: number of points for continuous predictor
  # ci_level: credible interval
  # integrated: pass to proj_linpred
  # seed: reproducibility
  
  set.seed(seed)
  
  # Determine if predictor is continuous or categorical
  if(is.numeric(newdata_template[[predictor]])) {
    x_grid <- seq(min(newdata_template[[predictor]], na.rm = TRUE),
                  max(newdata_template[[predictor]], na.rm = TRUE),
                  length.out = grid_length)
  } else {
    x_grid <- unique(newdata_template[[predictor]])
  }
  
  # Create newdata grid with all other predictors fixed at mean/mode
  X_base <- newdata_template %>%
    summarise(across(everything(), \(x) if(is.numeric(x)) mean(x, na.rm = TRUE) else Mode(x)))
  

  
  newdata_list <- lapply(x_grid, function(x) {
    df <- X_base
    df[[predictor]] <- x
    df
  })
  
  newdata_grid <- bind_rows(newdata_list)
  
  # Posterior predictions
  linpred <- proj_linpred(prjfit, newdata = newdata_grid, integrated = integrated)
  
  # Summarize
  posterior_mean <- colMeans(linpred$pred)
  ci_lower <- apply(linpred$pred, 2, quantile, (1 - ci_level)/2)
  ci_upper <- apply(linpred$pred, 2, quantile, 1 - (1 - ci_level)/2)
  
  plot_df <- data.frame(
    predictor_value = x_grid,
    mean = posterior_mean,
    lower = ci_lower,
    upper = ci_upper
  )
  
  return(plot_df)
}
