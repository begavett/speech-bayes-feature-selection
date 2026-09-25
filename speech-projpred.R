library(pacman)
p_load(dplyr, magrittr, ggplot2, psych, hablar, readr, tidyr, data.table, sjmisc,
       brms, loo, posterior, projpred, recipes, yardstick, future, doFuture)

source("src/CCF_cleaning_script.R")

train_child_min <- train_child %>%
  select(id, impaired, TOTAL_WORDS:epistemic_ratio, aget89, education, female, race) %>%
  filter(!race == "Native American",
         !is.na(race)) %>%
  mutate(race = factor(race) %>%
           relevel(ref = "Black"))

test_child_min <- test_child %>%
  select(id, impaired, TOTAL_WORDS:epistemic_ratio, aget89, education, female, race) %>%
  filter(!race == "Native American",
         !is.na(race)) %>%
  mutate(race = factor(race) %>%
           relevel(ref = "Black"))

projpred_rec <- recipe(impaired ~ ., data = train_child_min) %>%
  update_role(id, new_role = "ID") %>%
  step_nzv(all_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_dummy(race, one_hot = FALSE) %>%
  step_impute_bag(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

projpred_prep <- prep(projpred_rec, training = train_child_min)
projpred_prep

training_data_bk <- bake(projpred_prep, new_data = NULL) %>%
  select(-id)

describe(training_data_bk)

paste0(names(training_data_bk), collapse = " + ")

refmod_f <- formula(impaired ~ 
                      (aget89*female + education + race_Asian + race_LatinX + race_White) * 
                      (TOTAL_WORDS + NUM_SENTENCES + ADJ_TOTAL_WORDS_RATIO + 
                         ADV_TOTAL_WORDS_RATIO + ADP_TOTAL_WORDS_RATIO + 
                         CCONJ_TOTAL_WORDS_RATIO + DEM_TOTAL_WORDS_RATIO + 
                         NOUN_TOTAL_WORDS_RATIO + PRON_TOTAL_WORDS_RATIO +
                         VERB_TOTAL_WORDS_RATIO + CONTENT_FUNCTION_RATIO + 
                         NOUN_PRON_RATIO + NOUN_VERB_RATIO + MATTR_MEAN + 
                         NOUNS_ACQ_MEAN + NOUNS_AMB_MEAN + NOUNS_CONCRETE_MEAN + 
                         NOUNS_FAM_PREV_MEAN + NOUNS_FREQ_MEAN + PTAN_NOUNS + 
                         AROUSAL_NOUNS + VALENCE_NOUNS + INTJ_TOTAL_WORDS_RATIO + 
                         TENSE_PRESENT_PAST_RATIO + CPP + Intensity + 
                         Mean_f0 + Sd_f0 + Max_F0 + speakingrate + 
                         articulationrate + npause + asd + pause_ratio + 
                         mean_pause_duration + hedges_ratio + epistemic_ratio))

# Number of regression coefficients:
D <- refmod_f %>%
  model.matrix(data = training_data_bk) %>%
  ncol() %>%
  subtract(1)

D

# Prior guess for the number of relevant (i.e., non-zero) regression
# coefficients:
p0 <- 15
# Number of observations:
N <- sum(!is.na(training_data_bk$impaired))
# Hyperprior scale for tau, the global shrinkage parameter (note that for the
# Gaussian family, 'rstanarm' will automatically scale this by the residual
# standard deviation):

param_ratio <- p0 / (D - p0)
param_ratio



ref_model_hs <- brm(refmod_f,
                 data = training_data_bk,
                 chains = 4,
                 cores = 4,
                 #iter = 8000,
                 seed = 6,
                 save_pars = save_pars(all = TRUE),
                 file = "Analysis/ref_model_hs",
                 file_refit =  "on_change",
                 #threads = threading(threads = 4),
                 control = list(adapt_delta = .999,
                                max_treedepth = 15),
                 #backend = "cmdstanr",
                 # prior = prior(normal(0, 2), class = "b")
                 prior = prior(horseshoe(par_ratio = param_ratio), class = "b"),
                 family = "bernoulli")


ref_model_r2d2 <- brm(refmod_f,
                 data = training_data_bk,
                 chains = 4,
                 cores = 4,
                 #iter = 8000,
                 seed = 6,
                 save_pars = save_pars(all = TRUE),
                 file = "Analysis/ref_model_r2d2",
                 file_refit =  "on_change",
                 #threads = threading(threads = 4),
                 control = list(adapt_delta = .999,
                                max_treedepth = 15),
                 #backend = "cmdstanr",
                 # prior = prior(normal(0, 2), class = "b")
                 prior = prior(R2D2(), class = "b"),
                 family = "bernoulli")



ref_model_nml <- brm(refmod_f,
                      data = training_data_bk,
                      chains = 4,
                      cores = 4,
                      #iter = 8000,
                      seed = 6,
                      save_pars = save_pars(all = TRUE),
                      file = "Analysis/ref_model_nml",
                      file_refit =  "on_change",
                      #threads = threading(threads = 4),
                      control = list(adapt_delta = .999,
                                     max_treedepth = 15),
                      #backend = "cmdstanr",
                      prior = prior(normal(0, 1), class = "b"),
                      family = "bernoulli")

loo(ref_model_hs, ref_model_r2d2, ref_model_nml)

ref_model <- ref_model_r2d2

ref_model
bayes_R2(ref_model)
pp_check(ref_model, ndraws = 200)
loo(ref_model)

training_data_bk_check <- predict(ref_model) %>%
  data.frame() %>%
  mutate(.pred_impaired = case_when(Estimate > .5 ~ 1,
                                   Estimate <= .5 ~ 0) %>%
           factor(levels = 0:1, labels = c("0", "1"))) %>%
  bind_cols(training_data_bk)


training_data_bk_check %>%
  conf_mat(impaired, .pred_impaired) %>%
  summary()


refm_obj <- get_refmodel(ref_model)

n_predictors <- (fixef(ref_model) %>% nrow()) - 1

rownames(fixef(ref_model)) %>%
  dput()

if(!file.exists("Analysis/cvvs1_impaired.Rds")) {
  
  # Warning: if modeling interactions, the number of parameters can be > 300. This will take a long time and use a lot of RAM - plan accordingly
  
  options(future.globals.maxSize= 15000*1024^2)
  # For running projpred's CV in parallel (see cv_varsel()'s argument `parallel`):
  doFuture::registerDoFuture()
  future::plan(future::multisession, workers = 6) 
  progressr::handlers(global = TRUE)
  # Final cv_varsel() run:
  cvvs1_impaired <- cv_varsel(
    refm_obj,
    cv_method = "LOO",
    parallel = TRUE,
    nterms_max = 50,
    refit_prj = FALSE)
  # Tear down the CV parallelization setup:
  future::plan(future::sequential)
  closeAllConnections()
  # stopCluster(cl)
  # doParallel::stopImplicitCluster()
  # foreach::registerDoSEQ()
  
  saveRDS(cvvs1_impaired, "Analysis/cvvs1_impaired.Rds")
  
} else {
  cvvs1_impaired <- readRDS("Analysis/cvvs1_impaired.Rds")
}
options(scipen = 999)
plot(cvvs1_impaired, stats = c("rmse", "R2"), ranking_nterms_max = NA, deltas = FALSE, alpha = .05, size_position = "primary_x_bottom")
plot(cvvs1_impaired, stats = c("rmse", "R2"), ranking_nterms_max = NA, deltas = TRUE, alpha = .05,  size_position = "primary_x_bottom")
plot(cvvs1_impaired, stats = c("mlpd", "elpd"), ranking_nterms_max = NA, deltas = TRUE,  size_position = "primary_x_bottom")
suggest_size(cvvs1_impaired)

smmry <- summary(cvvs1_impaired, stats = "elpd", type = c("mean", "lower", "upper"),
                 deltas = TRUE)
print(smmry, digits = 1)





if(!file.exists("Analysis/cvvs2_impaired.Rds")) {
  options(future.globals.maxSize= 15000*1024^2)

  doFuture::registerDoFuture()
  future::plan(future::multisession, workers = 16)
  progressr::handlers(global = TRUE)
  # Final cv_varsel() run:
  cvvs2_impaired <- cv_varsel(
    refm_obj,
    cv_method = "LOO",
    parallel = TRUE,
    nterms_max = 21)
  future::plan(future::sequential)
  closeAllConnections()
  # stopCluster(cl)
  # doParallel::stopImplicitCluster()
  # foreach::registerDoSEQ()
  
  saveRDS(cvvs2_impaired, "Analysis/cvvs2_impaired.Rds")
} else {
  cvvs2_impaired <- readRDS("Analysis/cvvs2_impaired.Rds")
}

plot(cvvs2_impaired, stats = c("mlpd", "elpd", "rmse", "R2"), ranking_nterms_max = NA, size_position = "primary_x_bottom") + xlim(0, 21)

suggest_size(cvvs2_impaired)

smmry <- summary(cvvs2_impaired, stats = "elpd", type = c("mean", "lower", "upper"),
                 deltas = TRUE)
print(smmry, digits = 2)

impaired2_stab <- smmry$perf_sub %>%
  transmute(submodel_size = size, 
            Predictor = ranking_fulldata,
            ELPD = round(elpd, 2),
            ELPD_lower = round(elpd.lower, 2),
            ELPD_upper = round(elpd.upper, 2))

size_decided <- 19

rk <- ranking(cvvs2_impaired)
( pr_rk <- cv_proportions(rk) )
rk[["fulldata"]]
plot(pr_rk)


( predictors_final <- head(rk[["fulldata"]], size_decided) ) %>%
  dput()

plot(cv_proportions(rk, cumulate = TRUE))


# Post-selection inference ------------------------------------------------

prj <- project(
  refm_obj,
  predictor_terms = predictors_final, 
  ndraws = 4000, 
  seed = 94)
prj_mat <- as.matrix(prj)

source("src/conditional_effects_proj.R")
conditional_effects_proj(prjfit = prj, newdata_template = training_data_bk %>%
                           select(impaired, any_of(predictors_final)) %>%
                           na.omit(),
                         predictor = "NOUN_PRON_RATIO") %>%
  ggplot(aes(x = predictor_value, y = mean)) +
  geom_line() +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
  labs(x = "NOUN_PRON_RATIO", y = "Predicted outcome")

prj_drws <- as_draws_matrix(prj_mat)
prj_smmry <- summarize_draws(
  prj_drws,
  "median", "sd", p_above0 = ~ mean(.x > 0), p_below0 = ~ mean(.x < 0),
  function(x) quantile(x, probs = c(0.025, 0.975))
)
# Coerce to a `data.frame` because pkgdown versions > 1.6.1 don't print the
# tibble correctly:
prj_smmry %>% arrange(desc(abs(median))) %>% print(n = size_decided + 2)

prj_smmry %>% mutate(OR = exp(median)) %>% arrange(desc(abs(median))) %>% print(n = size_decided + 2)


prj_smmry <- prj_smmry %>%
  mutate(CrI = paste0("[", round(`2.5%`, 3), ", ", round(`97.5%`, 3), "]"),
         postprob = paste0(case_when(median < 0 ~ round(100*p_below0, 1),
                                     median > 0 ~ round(100*p_above0, 1)), "%")) %>%
  print(n = size_decided + 2) 



bayesplot_theme_set(ggplot2::theme_bw())
mcmc_intervals(prj_mat, regex_pars = "^b_")
mcmc_areas(prj_mat, regex_pars = "^b_", prob = .95)
