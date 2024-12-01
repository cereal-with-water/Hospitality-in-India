import pandas as pd
import numpy as np

# File paths
file1_path = 'C:/Users/aiden/Desktop/project/gdp_Punjab1.csv'
file2_path = 'C:/Users/aiden/Desktop/project/gdp_Punjab2.csv'
file3_path = 'C:/Users/aiden/Desktop/project/admission_data.csv'
file4_path = 'C:/Users/aiden/Desktop/project/pollution_data.csv'
file5_path = 'C:/Users/aiden/Desktop/project/mortality_data.csv'

# Read files
gdp_punjab1 = pd.read_csv(file1_path)
gdp_punjab2 = pd.read_csv(file2_path)
admission_data = pd.read_csv(file3_path)
pollution_data = pd.read_csv(file4_path)
mortality_data = pd.read_csv(file5_path)

# GDP data cleaning
def clean_gdp_data(df):
    df = df.replace(',', '', regex=True)  # Remove commas from numeric data
    if 'Description' in df.columns:
        df = df.drop(['Description'], axis=1)  # Remove unnecessary columns
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')  # Convert GDP data to numeric
    df = df.melt(id_vars=['Year'], var_name='District', value_name='GDP')  # Reshape data
    df['LOCATION'] = 'Punjab'  # Add LOCATION column
    # Convert Year column to integer type
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    return df

gdp_punjab1_cleaned = clean_gdp_data(gdp_punjab1)
gdp_punjab2_cleaned = clean_gdp_data(gdp_punjab2)

# Combine and aggregate GDP data
gdp_data_combined = pd.concat([gdp_punjab1_cleaned, gdp_punjab2_cleaned], ignore_index=True)
gdp_data_aggregated = gdp_data_combined.groupby(['LOCATION', 'Year']).agg({'GDP': 'sum'}).reset_index()

# Admission data cleaning
admission_data['D.O.A'] = pd.to_datetime(admission_data['D.O.A'], errors='coerce')
admission_data['D.O.D'] = pd.to_datetime(admission_data['D.O.D'], errors='coerce')
admission_data['DURATION OF STAY'] = pd.to_numeric(admission_data['DURATION OF STAY'], errors='coerce')
admission_data_cleaned = admission_data.dropna(subset=['D.O.A', 'D.O.D', 'DURATION OF STAY']).copy()
admission_data_cleaned['LOCATION'] = 'Punjab'  # Add LOCATION column
admission_data_cleaned = admission_data_cleaned[['LOCATION', 'D.O.A', 'D.O.D', 'AGE', 'GENDER', 'DURATION OF STAY']]

# Pollution data cleaning
pollution_data['DATE'] = pd.to_datetime(pollution_data['DATE'], errors='coerce')
pollution_data_cleaned = pollution_data.dropna(subset=['DATE']).copy()
pollution_data_cleaned['LOCATION'] = 'Punjab'  # Add LOCATION column
pollution_data_cleaned = pollution_data_cleaned[['LOCATION', 'DATE', 'AQI', 'PM2.5 AVG', 'PM10 AVG', 'NO2 AVG']]

# Mortality data cleaning
mortality_data['DATE OF BROUGHT DEAD'] = pd.to_datetime(mortality_data['DATE OF BROUGHT DEAD'], errors='coerce')
mortality_data['AGE'] = pd.to_numeric(mortality_data['AGE'], errors='coerce')
mortality_data_cleaned = mortality_data.dropna(subset=['DATE OF BROUGHT DEAD', 'AGE']).copy()
mortality_data_cleaned['LOCATION'] = 'Punjab'  # Add LOCATION column
mortality_data_cleaned = mortality_data_cleaned[['LOCATION', 'DATE OF BROUGHT DEAD', 'AGE', 'GENDER']]

# Check columns before merging
print("pollution_data_cleaned columns:", pollution_data_cleaned.columns)
print("mortality_data_cleaned columns:", mortality_data_cleaned.columns)
print("admission_data_cleaned columns:", admission_data_cleaned.columns)

# Step 1: Merge pollution data with mortality data
merged_data = pd.merge(
    pollution_data_cleaned,
    mortality_data_cleaned,
    left_on=['DATE', 'LOCATION'],
    right_on=['DATE OF BROUGHT DEAD', 'LOCATION'],
    how='inner'
)

# Drop duplicate date column from mortality data
merged_data = merged_data.drop('DATE OF BROUGHT DEAD', axis=1)

# Step 2: Aggregate admission data
# Group admission data by LOCATION and DATE to create a unique key combination
admission_data_cleaned['DATE'] = admission_data_cleaned['D.O.A'].dt.date
admission_aggregated = admission_data_cleaned.groupby(['LOCATION', 'DATE']).agg({
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
    on=['DATE', 'LOCATION'],
    how='inner'
)

# Step 4: Merge with GDP data
# Extract year from DATE
merged_data['Year'] = merged_data['DATE'].dt.year.astype(int)

# Convert Year column in GDP data to integer
gdp_data_aggregated['Year'] = pd.to_numeric(gdp_data_aggregated['Year'], errors='coerce').astype(int)

final_data = pd.merge(
    merged_data,
    gdp_data_aggregated,
    on=['LOCATION', 'Year'],
    how='left'  # Retain all records and use NaN if GDP data is missing
)

# Save final data
final_data.to_csv('C:/Users/aiden/Desktop/project/final_data.csv', index=False)

print("Data cleaning and merging complete. Final dataset saved to 'final_data.csv'.")
