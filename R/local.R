# R Studio API Code
library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Libraries
library(tidyverse)
library(Hmisc)
library(caret)
library(microbenchmark)
library(parallel)
library(doParallel)

# Data Import and Cleaning
gss_tbl <- spss.get("../data/GSS2006.sav") %>%
  select(starts_with("BIG5"), "HEALTH")  %>%
  mutate_all(.funs = function(x) ifelse(x == "<NA>", NA, x)) %>%
  # Rows missing 10 responses are missing all predictors
  # Rows missing 11 responses are missing all predictors and response
  # Want all the rows that do not have 10 or 11 responses missing
  filter(!(rowSums(is.na(.)) %in% c(10,11))) %>%
  mutate_all(as.numeric) %>%
  as_tibble() 

# Analysis

# preprocessing of data - impute any missing values
preprocess <- preProcess(gss_tbl, method = "knnImpute")
imputed_tbl <- predict(preprocess, gss_tbl)

# create 10 folds so that folds will be the same for all methods used
folds <-  createFolds(imputed_tbl$HEALTH, 10)

# Time model without parallelizing code
exec_time_np <- system.time({
  
  # Run extreme gradient boosting regression and compute 10-fold cv statistics
  xgb_model <- train(HEALTH ~ .*.*.,
                   data = imputed_tbl,
                   method = "xgbLinear",
                   tuneLength = 2,
                   trControl = trainControl(method = "cv",
                                            indexOut = folds,
                                            verboseIter = T)
  )
  
})

# Time model after parallelizing code
exec_time_p <- system.time({
  
  # Create cores to use in parallelization
  local_cluster <- makeCluster(detectCores() - 1)
  registerDoParallel(local_cluster)
  
  # Run extreme gradient boosting regression and compute 10-fold cv statistics
  xgb_model <- train(HEALTH ~ .*.*.,
                     data = imputed_tbl,
                     method = "xgbLinear",
                     tuneLength = 2,
                     trControl = trainControl(method = "cv",
                                              indexOut = folds,
                                              verboseIter = T)
  )
  
  # Stop cluster
  stopCluster(local_cluster)
  registerDoSEQ()
  
})

# Nonparallelized time
exec_time_np

# Parallelized time
exec_time_p

# Difference
exec_time_np - exec_time_p

# Non-parallelized code took 140.4 seconds on 1 core.
# Parallelized code took 39.101 seconds on 7 cores.
# Difference of 101.299 seconds