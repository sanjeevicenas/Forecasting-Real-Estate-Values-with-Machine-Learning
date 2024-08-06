# Databricks notebook source
# MAGIC %md
# MAGIC ## Understand the Structure of the Data
# MAGIC
# MAGIC **Area_Type**:   The Type of Area of Property
# MAGIC
# MAGIC **Availability**: Earliest time to move in the property, availability for possession.
# MAGIC
# MAGIC **Location**: Locality or Area in the city
# MAGIC
# MAGIC **Size**: Property Type (Like 3BHK, 4BHK)
# MAGIC
# MAGIC **Society**: The property in the society or not
# MAGIC
# MAGIC **Total Sqft area**: Area of property
# MAGIC
# MAGIC **Bathroom Nos**: No of Bathroom in that particular Property
# MAGIC
# MAGIC **Balcony**: No of Balcony
# MAGIC
# MAGIC **Price**: Price of the property (target Column)

# COMMAND ----------

# To read a csv file
import pandas as pd 

df = pd.read_csv('/dbfs/FileStore/tables/Property_Valuation_Data.csv',encoding='ISO-8859-1')
display(df)

# COMMAND ----------

# To read the first 5 rows
df.head()

# COMMAND ----------

# To get the data type of each column
df.info()

# COMMAND ----------

# To get the number of null values in each column
df.isnull().sum()

# COMMAND ----------


