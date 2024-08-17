# Databricks notebook source
# MAGIC %md
# MAGIC ## Understand the Structure of the Data
# MAGIC
# MAGIC **Area_Type**:   The Type of Area of Property
# MAGIC
# MAGIC **Availablity**: Earliest time to move in the property, availability for possession.
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

# MAGIC %md
# MAGIC
# MAGIC # Data Cleaning

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

df_copy_1 = df.copy()
# print(df_copy_1)

Availability_col = df_copy_1['availability'].to_string()
def convert_to_date_and_encode(value):
    if value == 'Ready To Move':
        return 1
    elif value == 'Immediate Possession':
        return 2
    try:
        # Attempt to convert to datetime
        date_value = pd.to_datetime(value + '-2024', format='%d-%b-%Y')
        # Format as 'YYYY-MM-DD'
        return date_value.strftime('%Y-%m-%d')
    except ValueError:
        # If conversion fails, return the original text
        return value
    
Availability_col = df_copy_1['availability'].apply(convert_to_date_and_encode)

print(Availability_col)

# COMMAND ----------

df_copy_2 = df.copy()

location_col = df_copy_2['location']
location_col.fillna('Unknown',inplace=True)
print(location_col)
print(location_col.isnull().sum())

# COMMAND ----------

df_copy_3 = df.copy()

size_col = df_copy_3['size']
size_col.fillna('Unknown',inplace=True)
print(size_col)
print(size_col.isnull().sum())

# COMMAND ----------

df_copy_4 = df.copy()

society_col = df_copy_4['society']
society_col.fillna('Unknown',inplace=True)
print(society_col)
print(society_col.isnull().sum())

# COMMAND ----------

df_copy_5 = df.copy()

total_sqft_col = df_copy_5['total_sqft']

total_sqft_col_li = []

for value in total_sqft_col:
    try:
        # Handling ranges with a hyphen
        if '-' in value:
            lower, upper = value.split('-')
            add_up_low = (float(lower.strip()) + float(upper.strip())) / 2
            total_sqft_col_li.append(add_up_low)
        
        # Handling square meters
        elif 'Sq. Meter' in value:
            sqm_value = float(value.replace('Sq. Meter', '').strip())
            sqft_m_value = sqm_value * 10.764
            total_sqft_col_li.append(sqft_m_value)
        
        # Handling square yards
        elif 'Sq. Yards' in value:
            sqy_value = float(value.replace('Sq. Yards', '').strip())
            sqft_y_value = sqy_value * 9.0
            total_sqft_col_li.append(sqft_y_value)
        
        # Handling acres
        elif 'Acres' in value:
            acre_value = float(value.replace('Acres', '').strip())
            sqft_acre_value = acre_value * 43560
            total_sqft_col_li.append(sqft_acre_value)
        
        # Handling gunthas
        elif 'Guntha' in value:
            guntha_value = float(value.replace('Guntha', '').strip())
            sqft_guntha_value = guntha_value * 1089.0
            total_sqft_col_li.append(sqft_guntha_value)
        
        # Handling cents
        elif 'Cents' in value:
            cents_value = float(value.replace('Cents', '').strip())
            sqft_cents_value = cents_value * 435.56
            total_sqft_col_li.append(sqft_cents_value)
        
        # Handling grounds
        elif 'Grounds' in value:
            grounds_value = float(value.replace('Grounds', '').strip())
            sqft_grounds_value = grounds_value * 2400.35
            total_sqft_col_li.append(sqft_grounds_value)
        
        # Handling perch
        elif 'Perch' in value:
            perch_value = float(value.replace('Perch', '').strip())
            sqft_perch_value = perch_value * 272.25
            total_sqft_col_li.append(sqft_perch_value)
        
        # Handling pure numeric values
        else:
            total_sqft_col_li.append(float(value.strip()))
    
    except ValueError as e:
        print(f"Error processing value: {value}, Error: {e}")
print(total_sqft_col_li)
print(len(total_sqft_col_li))
print(total_sqft_col_li.count(None))

# COMMAND ----------

df_copy_6 = df.copy()

bath_col = df_copy_6['bath']
bath_col.fillna(df_copy_6['bath'].median(),inplace=True)
print(bath_col)
print(bath_col.isnull().sum())
print(len(bath_col))

# COMMAND ----------

df_copy_7 = df.copy()

balcony_col = df_copy_7['balcony']
balcony_col.fillna(df_copy_7['balcony'].median(),inplace=True)
print(balcony_col)
print(balcony_col.isnull().sum())
print(len(balcony_col))

# COMMAND ----------

df_copy_8 = df.copy()
price_col = df_copy_8['price']

price_col_li = []
for value in price_col:
    try:
        if '$' in value:
            dollar_clean = float(value.replace('$', ''))
            convert_to_euro = dollar_clean * 0.92
            price_col_li.append(convert_to_euro)

        elif '-' in value:
            lower, upper = value.split('-')
            add_up_low = (float(lower.strip()) + float(upper.strip())) / 2
            price_col_li.append(add_up_low)

        elif '?' in value:
            q_clean = float(value.replace('?', ''))
            price_col_li.append(q_clean)
        
        elif '\x80' in value:
            euro_clean = float(value.replace('\x80', ''))
            price_col_li.append(euro_clean)
    
        else:
            price_col_li.append(float(value))
    except ValueError as e:
        print(f"Error processing value: {value}, Error: {e}")

print(price_col_li)
print(len(price_col_li))
print(df_copy_8['price'].isnull().sum())

# COMMAND ----------

area_type = pd.DataFrame(df['area_type'])
availability = pd.DataFrame(Availability_col)
location = pd.DataFrame(location_col)
size = pd.DataFrame(size_col)
society = pd.DataFrame(society_col)

cat_df = pd.concat([area_type, availability, location, size, society], axis=1)

display(cat_df)

# COMMAND ----------

price_col = price_col_li
price_col = pd.DataFrame(price_col, columns=['price'])
total_sqft_col = pd.DataFrame(total_sqft_col_li,columns=['total_sqft'])
bath = pd.DataFrame(bath_col)
balcony = pd.DataFrame(balcony_col)

# # Concatenate all DataFrames along the columns
num_df = pd.concat([price_col, total_sqft_col, bath, balcony], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# Encoding Categorical Variables
# Convert categorical variables into numerical values using techniques like one-hot encoding or label encoding.

required_columns = ['area_type', 'availability', 'location', 'size', 'society']

cat_df_encode = pd.get_dummies(cat_df, columns=required_columns)

display(cat_df_encode)

# COMMAND ----------

# Log Transformation
import numpy as np
# Apply log transformation to reduce the impact of outliers
num_df['price'] = np.log1p(num_df['price'])

display(num_df)

# COMMAND ----------

# Scaling and Normalization
# # Scale numerical features to a standard range.

from sklearn.preprocessing import StandardScaler

# Standardize the 'price' and 'total_sqft' columns
scaler = StandardScaler()
num_df[['price', 'total_sqft']] = scaler.fit_transform(num_df[['price', 'total_sqft']])

display(num_df)

# COMMAND ----------

# combine cat and numerical
cat_num_df = pd.concat([cat_df,num_df], axis = 1)

display(cat_num_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecasting ML model

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Select numerical columns
numerical_columns = ['total_sqft', 'bath', 'balcony', 'price_per_sqft']
X = num_df[numerical_columns]
y = num_df['price']

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
