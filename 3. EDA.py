# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sanjeevicenas84@gmail.com/Forecasting-Real-Estate-Values-with-Machine-Learning/Include"

# COMMAND ----------

# Descriptive Statistics

price_col.describe()

# COMMAND ----------

# Histogram

# To visualize the distribution of prices and identify skewness and outliers

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
sns.histplot(price_col['price'], bins=50, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
# plt.show()

# COMMAND ----------

# The histogram you've generated shows a heavily skewed distribution, which suggests that there may be issues with the data cleaning or that the data inherently has a few extremely high values (outliers) or a lot of low values.

# The histogram suggests that there may be significant outliers or a large number of very low values (close to zero). This is common in real estate data. You might want to explore these outliers further or consider applying a log transformation or capping outliers for a more interpretable distribution.

# COMMAND ----------

import numpy as np

# Log-transforming the price data
log_prices = np.log1p(price_col['price'])  # log1p handles log(0) by using log(1 + x)

# Plotting the histogram of log-transformed prices
plt.figure(figsize=(12, 6))
sns.histplot(log_prices, bins=50, kde=True)
plt.title('Log-Transformed Price Distribution')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
# plt.show()


# COMMAND ----------

# Interpretation:

# Normalization: The log transformation has normalized the price data, which can improve the performance of certain models, such as linear regression, that assume normally distributed data.
# Outliers: The distribution now shows fewer extreme outliers, making it easier to work with the data. You can further inspect the tails of the distribution to decide if any outliers need to be handled separately.

# COMMAND ----------

# Histogram

# To visualize the distribution of prices and identify skewness and outliers

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
sns.histplot(total_sqft_col['total_sqft'], bins=50, kde=True)
plt.title('Sqft Distribution')
plt.xlabel('Sqft')
plt.ylabel('Frequency')
# plt.show()

# COMMAND ----------

import numpy as np

# Log-transforming the price data
log_total_sqft = np.log1p(total_sqft_col['total_sqft'])  # log1p handles log(0) by using log(1 + x)

# Plotting the histogram of log-transformed prices
plt.figure(figsize=(12, 6))
sns.histplot(log_total_sqft, bins=50, kde=True)
plt.title('Log-Transformed sqft Distribution')
plt.xlabel('Log(Sqft)')
plt.ylabel('Frequency')
# plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the boxplot for log-transformed price data
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
sns.boxplot(x=log_prices)
plt.title('Boxplot of Log-Transformed Price')
plt.xlabel('Log(Price)')
# plt.show()

# COMMAND ----------

# Interpretation:

# Boxplot Components:
# The box in the boxplot represents the interquartile range (IQR), which contains the middle 50% of the data.

# The line inside the box represents the median of the log-transformed prices.

# The whiskers extend to 1.5 times the IQR from the box, and any points outside this range are considered outliers.

# Using the log-transformed data in a boxplot will give you a clearer view of the central tendency, variability, and potential outliers in your data, which may have been hidden or exaggerated in the original scale.

# COMMAND ----------

# Scatter Plot
# To visualize the relationship between 'price' and other numerical features like 'total_sqft'.

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(15, 10))
sns.scatterplot(x=log_total_sqft, y=log_prices)
plt.title('Price vs Total Sqft')
plt.xlabel('Total Sqft')
plt.ylabel('Price')
# plt.show()

# COMMAND ----------

# Pair Plot
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

log_bath = np.log1p(bath)
log_balcony = np.log1p(balcony)
num_df_pair = pd.concat([log_prices, log_total_sqft,log_balcony,log_bath],axis=1)


# Pair Plot
plt.figure(figsize=(15, 10))
sns.pairplot(num_df_pair)
# plt.show()
