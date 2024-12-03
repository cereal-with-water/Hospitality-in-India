import pandas as pd
import numpy as np

# Read files
admission_data = pd.read_csv('admission_data.csv')
pollution_data = pd.read_csv('pollution_data.csv')
mortality_data = pd.read_csv('mortality_data.csv')

# Admission data cleaning
admission_data['D.O.A'] = pd.to_datetime(admission_data['D.O.A'], errors='coerce')
admission_data['D.O.D'] = pd.to_datetime(admission_data['D.O.D'], errors='coerce')
admission_data['DURATION OF STAY'] = pd.to_numeric(admission_data['DURATION OF STAY'], errors='coerce')
admission_data_cleaned = admission_data.dropna(subset=['D.O.A', 'D.O.D', 'DURATION OF STAY']).copy()
admission_data_cleaned = admission_data_cleaned[['D.O.A', 'D.O.D', 'AGE', 'GENDER', 'DURATION OF STAY']]

# Pollution data cleaning
pollution_data['DATE'] = pd.to_datetime(pollution_data['DATE'], errors='coerce')
pollution_data_cleaned = pollution_data.dropna(subset=['DATE']).copy()
pollution_data_cleaned = pollution_data_cleaned[['DATE', 'AQI', 'PM2.5 AVG', 'PM10 AVG', 'NO2 AVG']]

# Mortality data cleaning
mortality_data['DATE OF BROUGHT DEAD'] = pd.to_datetime(mortality_data['DATE OF BROUGHT DEAD'], errors='coerce')
mortality_data['AGE'] = pd.to_numeric(mortality_data['AGE'], errors='coerce')
mortality_data_cleaned = mortality_data.dropna(subset=['DATE OF BROUGHT DEAD', 'AGE']).copy()
mortality_data_cleaned = mortality_data_cleaned[['DATE OF BROUGHT DEAD', 'AGE', 'GENDER']]

# Check columns before merging
print("pollution_data_cleaned columns:", pollution_data_cleaned.columns)
print("mortality_data_cleaned columns:", mortality_data_cleaned.columns)
print("admission_data_cleaned columns:", admission_data_cleaned.columns)

# Step 1: Merge pollution data with mortality data
merged_data = pd.merge(
    pollution_data_cleaned,
    mortality_data_cleaned,
    left_on=['DATE'],
    right_on=['DATE OF BROUGHT DEAD'],
    how='inner'
)

# Drop duplicate date column from mortality data
merged_data = merged_data.drop('DATE OF BROUGHT DEAD', axis=1)

# Step 2: Aggregate admission data
# Group admission data by DATE
admission_data_cleaned['DATE'] = admission_data_cleaned['D.O.A'].dt.date
admission_aggregated = admission_data_cleaned.groupby(['DATE']).agg({
    'AGE': 'mean',
    'GENDER': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    'DURATION OF STAY': 'mean'
}).reset_index()

# Convert DATE column to datetime format for merging
admission_aggregated['DATE'] = pd.to_datetime(admission_aggregated['DATE'])

# Step 3: Merge pollution and mortality data with admission data
merged_data = pd.merge(
    merged_data,
    admission_aggregated,
    on=['DATE'],
    how='inner'
)

# Step 4: Save final data
merged_data.to_csv('final_data.csv', index=False)
print("Data cleaning and merging complete. Final dataset saved to 'final_data.csv'.")
