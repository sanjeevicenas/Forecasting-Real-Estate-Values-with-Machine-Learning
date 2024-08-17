# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sanjeevicenas84@gmail.com/Forecasting-Real-Estate-Values-with-Machine-Learning/3. EDA"

# COMMAND ----------

# create a feature that represents the price per square foot.

num_df_pair['price_per_sqft'] = log_prices / log_total_sqft

display(num_df_pair['price_per_sqft'])

# COMMAND ----------

required_columns = ['location']

cat_df_encode = pd.get_dummies(cat_df, columns=required_columns)

cat_df_encode.drop(['availability','area_type','size','society'], axis=1, inplace=True)

display(cat_df_encode)

# COMMAND ----------

# Outlier Treatment

Q1 = num_df_pair['price'].quantile(0.25)
Q3 = num_df_pair['price'].quantile(0.75)
IQR = Q3-Q1

num_df_pair = num_df_pair[~((num_df_pair['price'] < (Q1 - 1.5 * IQR)) | (num_df_pair['price'] > (Q3 + 1.5 * IQR)))]

print(num_df_pair)

# COMMAND ----------

# Scaling and Normalization
# # Scale numerical features to a standard range.

from sklearn.preprocessing import StandardScaler

# Standardize the 'price' and 'total_sqft' columns
scaler = StandardScaler()
num_df_pair[['price', 'total_sqft']] = scaler.fit_transform(num_df_pair[['price', 'total_sqft']])
print(num_df_pair)

# COMMAND ----------

# combine cat and numerical
cat_num_df = pd.concat([cat_df_encode,num_df_pair], axis = 1)

cat_num_df.isnull().sum()

num_col = ['price', 'total_sqft','balcony','bath']

cleaned_df = cat_num_df.dropna(subset=num_col)

# COMMAND ----------

import pandas as pd
c_n_df = pd.concat([cat_df,num_df_pair], axis = 1)
c_n_df.isnull().sum()
clean=c_n_df.dropna(subset=num_col)
clean.isnull().sum()
clean

# COMMAND ----------

print(cleaned_df.isnull().sum())

# COMMAND ----------

print(cleaned_df)

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Select numerical columns
X = num_df_pair.drop(['price'], axis=1)
y = num_df_pair['price']

# COMMAND ----------

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)


# Predict on the test set
y_pred = lr.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

final_predictions = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})

display(final_predictions)

# COMMAND ----------

# Interpretation of the Metrics:
# MAE (Mean Absolute Error):

# MAE: 0.50 means that, on average, the model's predictions are off by about 0.5 units. This is a relatively low error, suggesting that the model is making fairly accurate predictions.

# MSE (Mean Squared Error):

# MSE: 0.45 represents the average of the squared errors. A lower MSE indicates that the model's predictions are generally close to the actual values.

# RMSE (Root Mean Squared Error):

# RMSE: 0.67 is the square root of MSE and provides an error metric in the same units as the target variable. An RMSE close to 0.67 suggests that the model is making predictions with a small average deviation.

# R-squared (R²):

# R-squared: 0.56 indicates that about 56% of the variance in the target variable is explained by the model. While this is a decent R² value, it suggests there is still room for improvement. A higher R² would indicate that the model explains more of the variance in the data.

# COMMAND ----------

# Cross Validation Model
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(lr, X, y, cv=5, scoring='r2')
print("Cross-validated R-squared scores:", cv_scores)
print("Average R-squared score:", np.mean(cv_scores))

# COMMAND ----------

#  Interpretation:
# The scores show some variability across different folds, which can happen with Linear Regression if there are nonlinear relationships in the data that the model isn’t capturing well.
# The average R-squared score of 0.5593 is lower than the Random Forest model’s score of 0.6547, which aligns with the expectation that more complex models like Random Forest often perform better on such tasks.

# COMMAND ----------

# Random Forest Regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

final_predictions = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})

display(final_predictions)

# COMMAND ----------

# Interpretation of the Metrics:
# MAE (Mean Absolute Error):

# MAE: 0.43 means that, on average, the model's predictions are off by about 0.43 units. This is a reduction from the previous model, indicating better accuracy.
# MSE (Mean Squared Error):

# MSE: 0.35 is lower than the previous model’s MSE, which suggests that the model's predictions are generally closer to the actual values.
# RMSE (Root Mean Squared Error):

# RMSE: 0.59 is also an improvement, indicating that the average prediction error is smaller.
# R-squared (R²):

# R-squared: 0.66 indicates that about 66% of the variance in the target variable is explained by the model. This is a significant improvement over the linear regression model, meaning the Random Forest model is capturing more of the complexity in the data.

# COMMAND ----------

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print("Cross-validated R-squared scores:", cv_scores)
print("Average R-squared score:", np.mean(cv_scores))

# COMMAND ----------

# Interpretation:
# The R-squared scores across the folds are fairly consistent, which is a positive sign. This consistency indicates that the model is reliable and not overly dependent on specific parts of the training data.
# The average R-squared score of 0.6547 aligns well with your initial training and testing results, reinforcing that the model is robust and performing as expected.

# COMMAND ----------

# Hyper Parameter Tuning

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_

# Evaluate the tuned model
y_pred = best_rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Tuned R-squared: {r2}")

# COMMAND ----------

# Interpretation:
# Improved R-squared: The tuned R-squared value of 0.676 is an improvement, indicating that the hyperparameter tuning has effectively enhanced the model's performance.

# COMMAND ----------

# Residuals

import matplotlib.pyplot as plt

residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Check for normal distribution of residuals
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.show()

# COMMAND ----------

# Interpretation:
# Ideally, the residuals should be randomly distributed around the horizontal line at zero, without any obvious pattern.
# In your plot, the residuals appear to be fairly scattered around zero, which is good. This suggests that the model's predictions are unbiased.
# However, there is a slight funnel shape where residuals spread more for predictions closer to zero and taper off as predictions increase or decrease. This could indicate heteroscedasticity, where the variance of the residuals is not constant.
