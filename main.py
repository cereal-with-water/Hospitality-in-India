import pandas as pd
import numpy as np

# load files (change path here!)
file1_path = 'gdp_Punjab1.csv'
file2_path = 'gdp_Punjab2.csv'
file3_path = 'admission_data.csv'
file4_path = 'pollution_data.csv'
file5_path = 'mortality_data.csv'

# read the files
gdp_punjab1 = pd.read_csv(file1_path)
gdp_punjab2 = pd.read_csv(file2_path)
admission_data = pd.read_csv(file3_path)
pollution_data = pd.read_csv(file4_path)
mortality_data = pd.read_csv(file5_path)

# GDP data cleaning
def clean_gdp_data(df):
    df = df.replace(',', '', regex=True)
    df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    return df
gdp_punjab1_cleaned = clean_gdp_data(gdp_punjab1)
gdp_punjab2_cleaned = clean_gdp_data(gdp_punjab2)

# admissions dataset cleaning
admission_data['D.O.A'] = pd.to_datetime(admission_data['D.O.A'], errors='coerce')
admission_data['D.O.D'] = pd.to_datetime(admission_data['D.O.D'], errors='coerce')
admission_data['DURATION OF STAY'] = pd.to_numeric(admission_data['DURATION OF STAY'], errors='coerce')
admission_data_cleaned = admission_data.dropna(subset=['D.O.A', 'D.O.D', 'DURATION OF STAY'])

#pollution dataset cleaning
pollution_data['DATE'] = pd.to_datetime(pollution_data['DATE'], errors='coerce')
pollution_data_cleaned = pollution_data.dropna(subset=['DATE'])

# mortality dataset cleaning
mortality_data['DATE OF BROUGHT DEAD'] = pd.to_datetime(mortality_data['DATE OF BROUGHT DEAD'], errors='coerce')
mortality_data['AGE'] = pd.to_numeric(mortality_data['AGE'], errors='coerce')
mortality_data['MRD'] = mortality_data['MRD'].replace('***', np.nan)
mortality_data_cleaned = mortality_data.dropna(subset=['DATE OF BROUGHT DEAD', 'AGE', 'MRD'])

# NEXT STEP: merge the datasets based on a common feature, in our case it would be the location of Punjab
# write code here to add a location column to all datasets

# merged the datasets by location
# gdp_data_combined = pd.concat([gdp_punjab1, gdp_punjab2], ignore_index=True)
# combined_gdp_mortality = pd.merge(gdp_data_combined, mortality_data_cleaned, on='LOCATION', how='inner')
# combined_all = pd.merge(combined_gdp_mortality, admission_data_cleaned, on='LOCATION', how='inner')
# combined_final = pd.merge(combined_all, pollution_data_cleaned, on='LOCATION', how='inner')





