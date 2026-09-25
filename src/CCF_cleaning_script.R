##cleaning script for workgroup 2 speech biomarker group

## load packages
library("tidyverse")   # Load tidyverse package
library("here") # load here package
library("ggplot2")
library("psych")

## load in raw data
dt <- read.csv(here::here("data/ucd_adc_language_deidentified.csv"))

## select childhood prompt and select only the first time point in which they completed the speech task
childhood<- dt %>% filter(prompt == "childhood") %>% group_by(id) %>% slice_min(order_by = time, n = 1)

## replace non-number variables with NA
childhood <- childhood %>%
  mutate(across(
    everything(),
    ~ replace(., . %in% c("inf", "-inf", "Inf", "-Inf", "-99"), NA)
  ))

## change character variable into numeric
childhood$TENSE_PRESENT_PAST_RATIO<- as.numeric(childhood$TENSE_PRESENT_PAST_RATIO)

##remove participants who do not have at least 50 words in this prompt
childhood <- childhood %>%
  filter(TOTAL_WORDS > 50)

##calculate ratios for hedging and epistemic variables and delete the original variables
childhood$hedges_ratio <- (childhood$hedges_count_LLM/(childhood$TOTAL_WORDS))
childhood$epistemic_ratio <- (childhood$epistemic_uncertainty_count_LLM/(childhood$TOTAL_WORDS))
childhood$epistemic_uncertainty_count_LLM<-NULL
childhood$hedges_count_LLM<-NULL

## pivot data from wide format to long format to do easier calculations across all variables at once
df_long <- childhood %>%
  pivot_longer(
    cols = 6:45,
    names_to = "variable",
    values_to = "value"
  )

##calculate mean and 5 sds above and below
means5 <- df_long %>%
  group_by(variable) %>%
  summarise(
    mean_value = mean(value, na.rm = TRUE),
    sd_value   = sd(value, na.rm = TRUE),
    lower_cut   = mean_value - 5 * sd_value,
    upper_cut   = mean_value + 5 * sd_value,
    .groups = "drop"
  )

## flag cases where data is >5 sd above or below the mean 
df_flagged <- df_long %>%
  left_join(means5, by = "variable") %>%
  mutate(
    flag = case_when(
      value < lower_cut ~ "below_5sd",
      value > upper_cut ~ "above_5sd",
      TRUE ~ "within_range"
    )
  )

## create a list of flagged participants and create a df that has the participant ids and the variable they are flagged for
flag<- df_flagged %>% filter(flag != "within_range")
flag_ps<- flag %>% select(id, variable, flag)


##replace the flagged cases with NA
for (i in seq_len(nrow(flag_ps))) {
  row_id <- flag_ps$id[i]
  var_name <- flag_ps$variable[i]
  childhood[childhood$id == row_id, var_name] <- NA
}

## save a cleaned df
clean_childhood <- childhood

## save the cleaned df 
write.csv(clean_childhood, "data/clean_childhood.csv", row.names = FALSE)


## read in full data set
data2 <- read.csv(here::here("data/ucd_adrc_main_deidentified.csv"))
clean_childhood <- merge(clean_childhood, data2, by = c( 'id', 'time'))
#Stratified 70/30 train/test split code:
set.seed(123)
split_idx <- unlist(lapply(split(seq_len(nrow(clean_childhood)), clean_childhood$dx), function(idx) {
  sample(idx, size = floor(0.7 * length(idx)))
}))

train_child <- clean_childhood[split_idx, ]
test_child  <- clean_childhood[-split_idx, ]

#double checking numbers of each dx group in test and train sets:

table(train_child$dx)
table(test_child$dx)
prop.table(table(train_child$dx))
prop.table(table(test_child$dx))
here()


## create variable for impaired vs not in both test and training
train_child$impaired <- ifelse(train_child$dx == "MCI",1,
                                   ifelse(train_child$dx == "Dementia", 1,0
                                   ))

table(train_child$impaired)
train_child$impaired<-as.factor(train_child$impaired)
levels(train_child$impaired)


test_child$impaired <- ifelse(test_child$dx == "MCI",1,
                               ifelse(test_child$dx == "Dementia", 1,0
                               ))

table(test_child$impaired)
test_child$impaired<-as.factor(test_child$impaired)
levels(test_child$impaired)
