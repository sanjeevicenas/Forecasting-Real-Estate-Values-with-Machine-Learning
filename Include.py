# Databricks notebook source
import pandas as pd 

df = pd.read_csv('/dbfs/FileStore/tables/Property_Valuation_Data.csv',encoding='ISO-8859-1')

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

# COMMAND ----------

df_copy_2 = df.copy()

location_col = df_copy_2['location']
location_col.fillna('Unknown',inplace=True)

# COMMAND ----------

df_copy_3 = df.copy()

size_col = df_copy_3['size']
size_col.fillna('Unknown',inplace=True)

# COMMAND ----------

df_copy_4 = df.copy()

society_col = df_copy_4['society']
society_col.fillna('Unknown',inplace=True)

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

# COMMAND ----------

df_copy_6 = df.copy()

bath_col = df_copy_6['bath']
bath_col.fillna(df_copy_6['bath'].median(),inplace=True)

# COMMAND ----------

df_copy_7 = df.copy()

balcony_col = df_copy_7['balcony']
balcony_col.fillna(df_copy_7['balcony'].median(),inplace=True)

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

# COMMAND ----------

area_type = pd.DataFrame(df['area_type'])
availability = pd.DataFrame(Availability_col)
location = pd.DataFrame(location_col)
size = pd.DataFrame(size_col)
society = pd.DataFrame(society_col)

cat_df = pd.concat([area_type, availability, location, size, society], axis=1)

# COMMAND ----------

price_col = price_col_li
price_col = pd.DataFrame(price_col, columns=['price'])
total_sqft_col = pd.DataFrame(total_sqft_col_li,columns=['total_sqft'])
bath = pd.DataFrame(bath_col)
balcony = pd.DataFrame(balcony_col)

# # Concatenate all DataFrames along the columns
num_df = pd.concat([price_col, total_sqft_col, bath, balcony], axis=1)

# COMMAND ----------


