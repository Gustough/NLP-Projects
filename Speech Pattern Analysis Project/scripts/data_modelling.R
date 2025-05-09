library('tidyverse')
#FIRST model with Kendall-Tau Agreement Score

data <- read_csv("output.csv")
xs <- data.frame(scale(data[, -1]))

set.seed(1)
sample <- sample(1:100, 80)
training <- xs[sample,]
testing <- xs[-sample,]
model <- lm(AS ~ ., data=training)

summary(model, new_data = testing)

#SECOND model with classification Agreement Score

library(tidyverse)
library(nnet)      # Multinomial logistic regression
library(MASS)      # Stepwise selection with AIC
library(caret)     # Accuracy & confusion matrix
library(pscl)      # Pseudo R²

# Load data
data_2 <- read_csv("model_features_data_2.csv")

# Ensure AS is a factor (categorical target variable)
data_2$AS <- as.factor(data_2$AS)

# Select and scale only numeric columns
numeric_cols <- select_if(data_2, is.numeric)
xs <- data.frame(scale(numeric_cols))  # Scale numeric columns
xs$AS <- data_2$AS  # Add AS back after scaling

# Train-test split
set.seed(1)
sample <- sample(1:90, 75)
training <- xs[sample,]
testing <- xs[-sample,]

# Fit full multinomial regression model
full_model <- multinom(AS ~ ., data=training)

# Perform Stepwise Selection
stepwise_model <- stepAIC(full_model, direction="both", trace=FALSE)

# Print selected variables
print(summary(stepwise_model))

# Predict on test data
predictions <- predict(stepwise_model, testing)

# Compute accuracy & confusion matrix
conf_matrix <- confusionMatrix(predictions, testing$AS)
print(conf_matrix)
accuracy <- conf_matrix$overall['Accuracy']
print(paste("Overall Accuracy:", round(accuracy, 3)))


