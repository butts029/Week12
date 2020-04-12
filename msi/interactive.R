# Libraries
library(Hmisc)
library(caret)
library(parallel)
library(doParallel)

## removed tidyverse package to reduce number of libraries for msi

# Data Import and Cleaning 
## changed all operations to base R because faster than tidyverse and also to remove tidyverse library
gss_tbl <- spss.get("week12/GSS2006.sav") 
gss_tbl <- gss_tbl[, c("BIG5A1", "BIG5A2", "BIG5B1", "BIG5B2", "BIG5C1", "BIG5C2","BIG5D1", "BIG5D2", "BIG5E1", "BIG5E2", "HEALTH")]
# Rows missing 10 responses are missing all predictors
# Rows missing 11 responses are missing all predictors and response
# Want all the rows that do not have 10 or 11 responses missing
gss_tbl <- gss_tbl[!(rowSums(is.na(gss_tbl)) %in% c(10, 11)), ]
gss_tbl <- apply(gss_tbl, 2, function(x) as.numeric(factor(x)))

# Analysis

# preprocessing of data - impute any missing values
preprocess <- preProcess(gss_tbl, method = "knnImpute")
imputed_tbl <- predict(preprocess, gss_tbl)

# create 10 folds so that folds will be the same for all methods used
folds <-  createFolds(imputed_tbl[,"HEALTH"], 10)

# Time model without parallelizing code
exec_time_np <- system.time({
  
  # Run extreme gradient boosting regression and compute 10-fold cv statistics
  xgb_model <- train(HEALTH ~ .*.*.,
                     data = imputed_tbl,
                     method = "xgbLinear",
                     tuneLength = 3,
                     trControl = trainControl(method = "cv",
                                              indexOut = folds,
                                              verboseIter = T)
  )
  
})

# Time model after parallelizing code
exec_time_p <- system.time({
  
  # Create cores to use in parallelization
  local_cluster <- makeCluster(2)
  registerDoParallel(local_cluster)
  
  # Run extreme gradient boosting regression and compute 10-fold cv statistics
  xgb_model <- train(HEALTH ~ .*.*.,
                     data = imputed_tbl,
                     method = "xgbLinear",
                     tuneLength = 3,
                     trControl = trainControl(method = "cv",
                                              indexOut = folds,
                                              verboseIter = T)
  )
  
  # Stop cluster
  stopCluster(local_cluster)
  registerDoSEQ()
  
})

# save results to csv
write.csv(cbind(c("Non - Parallelized", "Parallelized"), c(exec_time_np[3], exec_time_p[3])), file = "interactive.csv")

# Non-parallelized code took 401.164 seconds on 1 core.
# Parallelized code took 204.191 seconds on 2 cores.
# Difference of 196.973 seconds