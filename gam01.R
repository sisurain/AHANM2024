# GAM with 100 components
library(mgcv)
rm(list = ls())

n <- 1000 # Assuming 'n' is 800 for this example, but you can set it to any number you need

# Initialize a matrix to hold the X variables
p <- 100 # p is the number of additive components, we ignore the global intercept term

X <- matrix(nrow = n, ncol = p)

# Fill the matrix with standard normal values
for(i in 1:p) {
  X[,i] <- rnorm(n)
}

# Calculate Y
coefficients <- seq(1, 2*p-1, by=2) # Generates a sequence from 1 to 199 with a step of 2
Y <- X %*% coefficients

# Generate the error term
error_term <- rnorm(n)

# Add the error term to Y
Y <- Y + error_term

set.seed(123) # For reproducibility

# Create a data frame from X for ease of use with formulas
df <- as.data.frame(X)
names(df) <- paste0("x", 1:p) # Naming the variables x1, x2, ..., x100

# Add Y to the data frame
df$Y <- Y

# Sample 80% of the data indices for training
train_indices <- sample(1:nrow(df), size = 0.8 * nrow(df))

# Split the data
train_data <- df[train_indices, ]
validation_data <- df[-train_indices, ]
# Range of k values to explore
k_values <- seq(7, 7, by = 1) # Example range; adjust as needed
bs <- "cr"

# Placeholder for storing RMSE for each k
rmse_values <- numeric(length(k_values))
ptm <- proc.time()
for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  # Dynamically construct the formula
  # Note: For demonstration, we simplify by applying the same 'k' to all variables, which might not be optimal in practice
  variable_terms <- paste0("s(x", 1:p, ", bs=", "bs", ", k=", k, ")", collapse = " + ")
  formula <- as.formula(paste("Y ~ ", variable_terms))
  
  # Fit the GAM model
  gam_model <- bam(formula, data=train_data, method="GCV.Cp")
  
  # Predict on validation data
  predictions <- predict(gam_model, newdata=validation_data)
  
  # Calculate RMSE
  rmse_values[i] <- sqrt(mean((as.vector(predictions) - validation_data$Y)^2))
}

# Find the k value that minimizes RMSE
optimal_k <- k_values[which.min(rmse_values)]
print(optimal_k)
proc.time() - ptm

