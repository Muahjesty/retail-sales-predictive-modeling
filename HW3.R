# ==============================================================================
# HW 3: Prediction - Linear Regression, rpart, Derived Attributes
# ==============================================================================

# Load needed libraries
library(rpart)

# 1. Load the data
df <- read.csv("retail_sales_dataset.csv")

# Data inspection and cleanup 
cols_to_use <- c("Total.Amount", "Gender", "Age", "Product.Category", "Quantity")
df <- df[complete.cases(df[, cols_to_use]), ]

# Make sure categorical variables are treated as factors
df$Gender <- as.factor(df$Gender)
df$Product.Category <- as.factor(df$Product.Category)


# ==============================================================================
# TASK 1 — Build Two Prediction Models (Baseline Definitions)
# ==============================================================================

# 1a. Numerical Variable — Linear Regression (lm)
# Target: Total.Amount | Predictors: Age, Gender, Product.Category, Quantity
formula_num <- Total.Amount ~ Age + Gender + Product.Category + Quantity
model_lm <- lm(formula_num, data = df)

# 1b. Categorical Variable — Decision Tree (rpart)
# Target: Gender | Predictors: Age, Total.Amount, Product.Category, Quantity
formula_cat <- Gender ~ Age + Total.Amount + Product.Category + Quantity
model_tree <- rpart(formula_cat, data = df, method = "class")


# ==============================================================================
# TASK 2 — Cross-Validation (Baseline)
# ==============================================================================
# I will run a 1-Fold cross-validation (70/30 train/test split) 5 times to 
# see the variance and stability of our models.

# Vectors to store the results of each run
rmse_baseline <- numeric(5)
acc_baseline <- numeric(5)

set.seed(123) # Set seed for reproducibility

for (i in 1:5) {
  # Split: 70% Training, 30% Testing
  train_indices <- sample(1:nrow(df), 0.7 * nrow(df))
  train_data <- df[train_indices, ]
  test_data  <- df[-train_indices, ]
  
  # --- Evaluate Linear Regression (Numerical) ---
  m_lm <- lm(formula_num, data = train_data)
  p_lm <- predict(m_lm, test_data)
  rmse_baseline[i] <- sqrt(mean((test_data$Total.Amount - p_lm)^2))
  
  # --- Evaluate rpart Decision Tree (Categorical) ---
  m_tr <- rpart(formula_cat, data = train_data, method = "class")
  p_tr <- predict(m_tr, test_data, type = "class")
  acc_baseline[i] <- mean(p_tr == test_data$Gender)
}

cat("--- TASK 2: BASELINE CV RESULTS ---\n")
cat("Baseline RMSE runs: ", rmse_baseline, "\n")
cat("Mean Baseline RMSE: ", mean(rmse_baseline), "\n\n")

cat("Baseline Accuracy runs: ", acc_baseline, "\n")
cat("Mean Baseline Accuracy: ", mean(acc_baseline), "\n\n")

# FINDINGS & REMARKS (Task 2):
# - Conclusions from multiple CV runs: By running the 1-fold CV 5 times, I can 
#   see the stability of the models. If the RMSE and Accuracy change and fluctuate a lot 
#   between runs, the model is more often than not dependent on the specific data split and is 
#   unstable and/or overfitting. If the metrics stay mostly tight and consistent 
#   across all 5 runs, I can be sure and confident that the mean error is a reliable 
#   estimate of real-world performance.


# ==============================================================================
# TASK 3 — Define Derived Attributes & Re-run CV
# ==============================================================================

# 1. Row-based Derived Attribute: 'Age_Squared'
# Computed using only values from the same row. I will square the age to help 
# the linear model get potential non-linear relationships between Age and Spend.
df$Age_Squared <- df$Age^2

# 2. Aggregated Derived Attribute: 'Avg.Spend.Category'
# Computed by grouping rows by a categorical variable and calculating a summary statistic.
# Here, I calculated the average Total.Amount per Product.Category.
agg_data <- aggregate(Total.Amount ~ Product.Category, data = df, FUN = mean)
colnames(agg_data)[2] <- "Avg.Spend.Category"
df <- merge(df, agg_data, by = "Product.Category")

# Define NEW Improved Formulas including the derived attributes
formula_num_new <- Total.Amount ~ Age + Age_Squared + Gender + Product.Category + Quantity + Avg.Spend.Category
formula_cat_new <- Gender ~ Age + Age_Squared + Total.Amount + Product.Category + Quantity + Avg.Spend.Category

# Vectors to store the results of the new CV runs
rmse_improved <- numeric(5)
acc_improved <- numeric(5)

set.seed(123) # Keep the same seed to make sure identical data splits for fair comparison

for (i in 1:5) {
  train_indices <- sample(1:nrow(df), 0.7 * nrow(df))
  train_data <- df[train_indices, ]
  test_data  <- df[-train_indices, ]
  
  # --- Evaluate Improved Linear Regression ---
  m_lm_new <- lm(formula_num_new, data = train_data)
  p_lm_new <- predict(m_lm_new, test_data)
  rmse_improved[i] <- sqrt(mean((test_data$Total.Amount - p_lm_new)^2))
  
  # --- Evaluate Improved rpart Decision Tree ---
  m_tr_new <- rpart(formula_cat_new, data = train_data, method = "class")
  p_tr_new <- predict(m_tr_new, test_data, type = "class")
  acc_improved[i] <- mean(p_tr_new == test_data$Gender)
}

cat("--- TASK 3: IMPROVED CV RESULTS ---\n")
cat("Improved Mean RMSE:     ", mean(rmse_improved), "\n")
cat("RMSE Improvement:       ", mean(rmse_baseline) - mean(rmse_improved), "\n\n")

cat("Improved Mean Accuracy: ", mean(acc_improved), "\n")
cat("Accuracy Improvement:   ", mean(acc_improved) - mean(acc_baseline), "\n\n")

# FINDINGS & REMARKS (Task 3):
# - Which derived attribute improved the accuracy/MSE: 
#   The aggregated attribute ('Avg.Spend.Category') mostly gives and provides the largest 
#   boost to the numerical model because it bakes the historical group average 
#   directly into the row, giving the linear model a high correlated baseline 
#   to adjust from. 'Age_Squared' acts as a feature engineering step that helps 
#   get the non-linear variance.
#
# - How to conclude the improvement is statistically significant:
#   I determined statistical significance by looking at the variance across the 5 CV 
#   folds. If the average improvement (for example, RMSE drops by 50) is hugely 
#   larger than the random fluctuation between the baseline CV runs (ex. the 
#   baseline runs only varied by +/- 5), the improvement is real and statistically 
#   significant. If the improvement is smaller than the natural variance between 
#   the folds, it is likely just statistical noise.

