# Load necessary library
library(mgcv)
library(glmnet)
rm(list = ls())

# Set parameters
n <- 1000
p <- 10
coefficients <- seq(1, 2*p-1, by=2)
#coefficients <- c(1, 3, rep(0, p-2))
n_iterations <- 100
rmse_values_gam <- numeric(n_iterations)
rmse_values_lm <- numeric(n_iterations)
rmse_values_lasso <- numeric(n_iterations)

for (i in 1:n_iterations) {
  
  # Generate data
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- 2 + X %*% coefficients + rnorm(n, mean = 0, sd = sqrt(0.2))
  
  # Split into training and testing sets
  train_indices <- sample(1:n, 800)
  test_indices <- setdiff(1:n, train_indices)
  
  X_train <- X[train_indices, ]
  y_train <- y[train_indices]
  X_test <- X[test_indices, ]
  y_test <- y[test_indices]
  
  # Convert to data frames for mgcv and lm
  train_data <- data.frame(y_train, X_train)
  names(train_data) <- c("y", paste0("X", 1:p))
  test_data <- data.frame(y_test, X_test)
  names(test_data) <- c("y", paste0("X", 1:p))
  
  # Fit GAM model with cubic splines
  formula_gam <- as.formula(paste("y ~", paste(paste0("s(X", 1:p, ", bs='cr')"), collapse=" + ")))
  gam_model <- gam(formula_gam, data=train_data)
  
  # Fit linear regression model
  formula_lm <- as.formula(paste("y ~", paste(paste0("X", 1:p), collapse=" + ")))
  lm_model <- lm(formula_lm, data=train_data)
  
  # Fit Lasso regression model
  lasso_model <- glmnet(as.matrix(X_train), y_train, alpha = 1)
  cv_model <- cv.glmnet(X_train, y_train, alpha = 1, type.measure="mse", nfolds=10)
  lambda_min_mse <- cv_model$lambda.min
  
  # Predict on testing set using GAM
  predictions_gam <- predict(gam_model, newdata=test_data)
  
  # Predict on testing set using linear regression
  predictions_lm <- predict(lm_model, newdata=test_data)
  
  # Predict on testing set using Lasso regression
  predictions_lasso <- predict(lasso_model, newx = as.matrix(X_test), s = lambda_min_mse)
  
  # Calculate RMSE for GAM
  #rmse_values_gam[i] <- sqrt(mean((predictions_gam - test_data$y)^2))
  rmse_values_gam[i] <- sum((predictions_gam - test_data$y)^2)
  
  # Calculate RMSE for linear regression
  #rmse_values_lm[i] <- sqrt(mean((predictions_lm - test_data$y)^2))
  rmse_values_lm[i] <- sum((predictions_lm - test_data$y)^2)
  
  # Calculate RMSE for Lasso regression
  #rmse_values_lasso[i] <- sqrt(mean((predictions_lasso - y_test)^2))
  rmse_values_lasso[i] <- sum((predictions_lasso - y_test)^2)
}

# Results
mean_rmse_gam <- mean(rmse_values_gam)
sd_rmse_gam <- sd(rmse_values_gam)
mean_rmse_lm <- mean(rmse_values_lm)
sd_rmse_lm <- sd(rmse_values_lm)
mean_rmse_lasso <- mean(rmse_values_lasso)
sd_rmse_lasso <- sd(rmse_values_lasso)

cat("GAM - Mean RMSE over 100 iterations: ", mean_rmse_gam, "\n")
cat("GAM - Standard deviation of RMSE over 100 iterations: ", sd_rmse_gam, "\n")
cat("Linear Regression - Mean RMSE over 100 iterations: ", mean_rmse_lm, "\n")
cat("Linear Regression - Standard deviation of RMSE over 100 iterations: ", sd_rmse_lm, "\n")
cat("Lasso Regression - Mean RMSE over 100 iterations: ", mean_rmse_lasso, "\n")
cat("Lasso Regression - Standard deviation of RMSE over 100 iterations: ", sd_rmse_lasso, "\n")

