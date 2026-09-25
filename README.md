# speech-bayes-feature-selection

Bayesian feature selection for speech-based classification of cognitive impairment, using regularized Bayesian logistic regression and projection predictive inference in R.

The analysis takes automatically derived acoustic and linguistic features from a spoken childhood-memory prompt and asks which of them (and which interactions with age, sex, education, and race) carry the most information for distinguishing cognitively impaired participants (MCI or dementia) from cognitively unimpaired participants. A full "reference model" is fit with shrinkage priors in [`brms`](https://paul-buerkner.github.io/brms/), and [`projpred`](https://mc-stan.org/projpred/) is then used to find a small submodel that predicts nearly as well.

## Repository structure

```
.
├── speech-projpred.R                 # Main analysis script
├── src/
│   ├── CCF_cleaning_script.R         # Data cleaning, outlier handling, train/test split
│   └── conditional_effects_proj.R    # Helper for conditional-effects curves from a projected model
├── .gitignore
└── README.md
```

The `data/` and `Analysis/` directories are git-ignored. You need to create `data/` and supply the input files yourself (see below). `Analysis/` is where fitted models and cached results are written.

## Data

The raw data are not included in this repository. The cleaning script expects two de-identified CSV files in a `data/` folder at the repository root:

| File | Contents |
|---|---|
| `data/ucd_adc_language_deidentified.csv` | Speech and language features, one row per participant, prompt, and time point |
| `data/ucd_adrc_main_deidentified.csv` | Clinical and demographic data, including diagnosis (`dx`), age, sex, education, and race |

The two files are merged on `id` and `time`. If you have access to the data, place the files in `data/` before running anything.

## Pipeline

### 1. Data cleaning (`src/CCF_cleaning_script.R`)

- Keeps only the `"childhood"` speech prompt and, for each participant, the earliest time point.
- Recodes `inf`, `-inf`, and `-99` to `NA`.
- Excludes participants with 50 or fewer words in the response.
- Converts LLM-derived hedge and epistemic-uncertainty counts to ratios of total words (`hedges_ratio`, `epistemic_ratio`).
- Sets values more than 5 SD from the feature mean to `NA` (per feature, not per participant).
- Merges in the clinical data and creates a stratified 70/30 train/test split by diagnosis (`set.seed(123)`).
- Defines the outcome `impaired` as 1 for MCI or dementia and 0 otherwise.

The script also writes `data/clean_childhood.csv`.

### 2. Preprocessing and reference model (`speech-projpred.R`)

- Removes participants with a missing or Native American race category, and uses Black as the race reference level.
- Applies a `recipes` pipeline fit on the training data: near-zero-variance filter, correlation filter, dummy coding of race, bagged-tree imputation, and normalization of numeric predictors.
- Fits a Bernoulli (logistic) `brms` model in which demographic terms (age, sex, education, race, and an age-by-sex interaction) are crossed with about 38 speech features. This gives a large number of coefficients, so shrinkage is essential.
- Compares three priors on the regression coefficients with LOO cross-validation:
  - Regularized horseshoe, with a prior guess of 15 relevant coefficients
  - R2D2
  - Normal(0, 1)
- Uses the R2D2 model as the reference model, and checks it with Bayesian R², posterior predictive checks, and a confusion matrix on the training data.

### 3. Projection predictive variable selection

- Runs `projpred::cv_varsel()` with LOO cross-validation and forward search, first up to 50 terms and then up to 21 terms.
- Plots predictive performance (ELPD, MLPD, RMSE, R²) against submodel size and uses `suggest_size()` as a guide.
- Fixes the final submodel size (19 terms in the current script) and inspects how stable the predictor ranking is with `cv_proportions()`.

### 4. Post-selection inference

- Projects the reference model onto the selected predictors (4,000 draws).
- Summarizes the projected coefficients with medians, SDs, 95% credible intervals, posterior probabilities of direction, and odds ratios.
- Plots coefficient intervals and areas with `bayesplot`.
- Draws conditional-effect curves for a chosen predictor with `conditional_effects_proj()`.

## Helper: `conditional_effects_proj()`

Computes a conditional-effects curve from a `projpred` projection. It varies one predictor over a grid (or over its levels, if categorical) while holding all other predictors at their mean (numeric) or mode (otherwise).

```r
source("src/conditional_effects_proj.R")

ce <- conditional_effects_proj(
  prjfit = prj,
  newdata_template = training_data_bk,
  predictor = "NOUN_PRON_RATIO",
  grid_length = 50,
  ci_level = 0.95
)
# Returns a data.frame: predictor_value, mean, lower, upper
```

Predictions are on the linear predictor scale (`proj_linpred()`).

## Requirements

R with the following packages. `pacman` is used to load most of them, and `tidyverse`, `here`, and `pacman` itself are loaded directly.

- Data handling: `tidyverse`, `dplyr`, `tidyr`, `readr`, `magrittr`, `data.table`, `hablar`, `sjmisc`, `here`, `pacman`
- Modeling: `brms` (with a working Stan backend), `loo`, `posterior`, `projpred`, `recipes`, `yardstick`
- Parallelization: `future`, `doFuture`, `progressr`
- Plotting and description: `ggplot2`, `psych`, `bayesplot`

Some recipe steps (`step_impute_bag`) may need `ipred`.

## Running the analysis

1. Clone the repository and open R at the repository root, so relative paths and `here()` resolve correctly.
2. Create the `data/` and `Analysis/` folders and add the two input CSVs to `data/`.
3. Run `speech-projpred.R`. It sources the two scripts in `src/` itself.

```r
dir.create("data"); dir.create("Analysis")
source("speech-projpred.R")
```

## Notes on computation

- The three `brms` models are fit with 4 chains, `adapt_delta = 0.999`, and `max_treedepth = 15`. Fits are cached to `Analysis/` and only refit when the model changes.
- `cv_varsel()` is slow and memory-hungry with this many interaction terms (over 300 coefficients). The script parallelizes it with `future::multisession` (6 and 16 workers) and raises `future.globals.maxSize` to roughly 15 GB. Adjust the worker count and memory limit to fit your machine.
- Results of `cv_varsel()` are cached as `.Rds` files in `Analysis/`. Delete them to rerun the search.
- The script does not set a `projpred`-specific seed for `cv_varsel()`, but it does seed the `brms` fits (`seed = 6`) and the final projection (`seed = 94`).

## Known quirks

- The cleaning script selects the features for outlier screening by column position (`cols = 6:45`), so it depends on the column order of the input file.
- A held-out test set (`test_child_min`) is created but not used in the current script.