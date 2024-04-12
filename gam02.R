#GAM with 4 components
library(mgcv)
rm(list = ls())

n <- 800 # Assuming 'n' is 800 for this example, but you can set it to any number you need

# Initialize a matrix to hold the X variables
p <- 4 # p is the number of additive components, we ignore the global intercept term
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

bs <- "cr"
k <- 7
# Placeholder for storing RMSE for each k

ptm <- proc.time()
b <- gam(Y ~ s(x1,bs=bs,k=k)+s(x2,bs=bs,k=k)+s(x3,bs=bs,k=k)+s(x4,bs=bs,k=k), data= train_data, method="GCV.Cp")
#b <- gam(Y ~ s(x1,bs=bs)+s(x2,bs=bs)+s(x3,bs=bs)+s(x4,bs=bs), data= train_data, method="GCV.Cp")
proc.time() - ptm

