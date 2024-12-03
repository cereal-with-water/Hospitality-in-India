import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import warnings

warnings.filterwarnings('ignore')

# 1. Load data
final_data = pd.read_csv('final_data.csv', parse_dates=['DATE'])

print("Data successfully loaded.")
print(final_data.head())
print(final_data.info())

# 2. Exploratory Data Analysis (EDA)

# 2.1 Check basic statistics
print("\nBasic Statistics:")
print(final_data.describe())

# 2.2 Check for missing values
print("\nMissing Values:")
print(final_data.isnull().sum())

# 2.3 Correlation Analysis
# Select numeric columns only
numeric_cols = final_data.select_dtypes(include=[np.number]).columns
corr_matrix = final_data[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 2.4 Visualize key variables

# Relationship between AQI and duration of stay
plt.figure(figsize=(8, 6))
sns.scatterplot(x='AQI', y='DURATION OF STAY', data=final_data)
plt.title('Relationship Between AQI and Duration of Stay')
plt.show()

# Trend of AQI over the years
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='AQI', data=final_data, marker='o')
plt.title('AQI Trend by Year')
plt.show()

# 3. Feature Engineering

# 3.1 Create features from the date
final_data['Month'] = final_data['DATE'].dt.month
final_data['Day'] = final_data['DATE'].dt.day
final_data['Weekday'] = final_data['DATE'].dt.weekday  # 0: Monday, 6: Sunday

# 3.2 Encode categorical variables
le_gender = LabelEncoder()

# Process 'GENDER_x' and 'GENDER_y'
if 'GENDER_x' in final_data.columns and 'GENDER_y' in final_data.columns:
    # Combine two GENDER columns
    final_data['GENDER'] = final_data['GENDER_x'].fillna(final_data['GENDER_y'])
    final_data['GENDER'] = final_data['GENDER'].fillna('Unknown')  # Handle missing values
    final_data['GENDER_ENCODED'] = le_gender.fit_transform(final_data['GENDER'])

    # Drop unnecessary columns
    final_data = final_data.drop(['GENDER_x', 'GENDER_y', 'GENDER'], axis=1)
elif 'GENDER_x' in final_data.columns:
    final_data['GENDER'] = final_data['GENDER_x'].fillna('Unknown')
    final_data['GENDER_ENCODED'] = le_gender.fit_transform(final_data['GENDER'])
    final_data = final_data.drop(['GENDER_x', 'GENDER'], axis=1)
elif 'GENDER_y' in final_data.columns:
    final_data['GENDER'] = final_data['GENDER_y'].fillna('Unknown')
    final_data['GENDER_ENCODED'] = le_gender.fit_transform(final_data['GENDER'])
    final_data = final_data.drop(['GENDER_y', 'GENDER'], axis=1)

# 3.3 Drop unnecessary variables
final_data = final_data.drop(['DATE', 'D.O.A', 'D.O.D', 'LOCATION'], axis=1, errors='ignore')

# 3.4 Handle missing values (if needed)
# If all values in GDP are NaN, drop the column; otherwise, fill with the mean
if final_data['GDP'].isnull().all():
    print("Warning: All values in 'GDP' are NaN. Dropping 'GDP' from modeling.")
    final_data = final_data.drop(['GDP'], axis=1)
else:
    final_data['GDP'] = final_data['GDP'].fillna(final_data['GDP'].mean())

print("\nData after preprocessing:")
print(final_data.info())

# 4. Define target variables

# **Classification example: 'admission_type' column not present**
print("Warning: 'admission_type' column is not available in the data.")
target_classification = None

# **Regression example: Predict duration of stay**
target_regression = 'DURATION OF STAY'

# 5. Split the data into training and test sets

# **Regression data preparation**
X_reg = final_data.drop([target_regression], axis=1, errors='ignore')
y_reg = final_data[target_regression]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 6. Model selection and training

print("\n=== Regression Task: Predicting Duration of Stay ===")

# Linear Regression
lr_regressor = LinearRegression()
lr_regressor.fit(X_train_reg, y_train_reg)
y_pred_lr_reg = lr_regressor.predict(X_test_reg)

print("\nLinear Regression Results:")
print("MSE:", mean_squared_error(y_test_reg, y_pred_lr_reg))
print("R²:", r2_score(y_test_reg, y_pred_lr_reg))

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)
y_pred_rf_reg = rf_regressor.predict(X_test_reg)

print("\nRandom Forest Regressor Results:")
print("MSE:", mean_squared_error(y_test_reg, y_pred_rf_reg))
print("R²:", r2_score(y_test_reg, y_pred_rf_reg))

# 7. Model evaluation and tuning

# **7.1 Cross-validation**

print("\n=== Regression Task: Cross-validation ===")
cv_scores_lr_reg = cross_val_score(lr_regressor, X_train_reg, y_train_reg, cv=5, scoring='neg_mean_squared_error')
print("Linear Regression CV MSE:", -cv_scores_lr_reg)
print("Mean CV MSE:", -cv_scores_lr_reg.mean())

cv_scores_rf_reg = cross_val_score(rf_regressor, X_train_reg, y_train_reg, cv=5, scoring='neg_mean_squared_error')
print("Random Forest Regressor CV MSE:", -cv_scores_rf_reg)
print("Mean CV MSE:", -cv_scores_rf_reg.mean())

# **7.2 Hyperparameter tuning (Random Forest example)**

print("\n=== Regression Task: Random Forest Hyperparameter Tuning ===")
param_grid_rf_reg = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search_rf_reg = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid_rf_reg,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search_rf_reg.fit(X_train_reg, y_train_reg)
print("Best Parameters:", grid_search_rf_reg.best_params_)
print("Best CV MSE:", -grid_search_rf_reg.best_score_)

# Predict with the best model
best_rf_regressor = grid_search_rf_reg.best_estimator_
y_pred_best_rf_reg = best_rf_regressor.predict(X_test_reg)

print("\nTuned Random Forest Regressor Results:")
print("MSE:", mean_squared_error(y_test_reg, y_pred_best_rf_reg))
print("R²:", r2_score(y_test_reg, y_pred_best_rf_reg))

# 8. Interpretation and insights

# **8.2 Regression Task: Feature Importance**
feature_importances_reg = best_rf_regressor.feature_importances_
feature_names_reg = X_train_reg.columns
feature_importances_series_reg = pd.Series(feature_importances_reg, index=feature_names_reg).sort_values(
    ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances_series_reg, y=feature_importances_series_reg.index)
plt.title('Feature Importances - Random Forest Regressor')
plt.show()

# **8.3 Visualize predictions (Regression Task)**
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_reg, y=y_pred_best_rf_reg)
plt.xlabel('Actual Duration of Stay')
plt.ylabel('Predicted Duration of Stay')
plt.title('Actual vs Predicted Duration of Stay')
plt.show()

# **8.4 Model Performance Summary**
print("\n=== Regression Model Performance Summary ===")
print("Linear Regression MSE:", mean_squared_error(y_test_reg, y_pred_lr_reg))
print("Linear Regression R²:", r2_score(y_test_reg, y_pred_lr_reg))
print("Random Forest Regressor MSE:", mean_squared_error(y_test_reg, y_pred_rf_reg))
print("Random Forest Regressor R²:", r2_score(y_test_reg, y_pred_rf_reg))
print("Tuned Random Forest Regressor MSE:", mean_squared_error(y_test_reg, y_pred_best_rf_reg))
print("Tuned Random Forest Regressor R²:", r2_score(y_test_reg, y_pred_best_rf_reg))

# 9. Save final model (optional)
# Example: Save the model using Pickle

import pickle

# Save regression model
with open('best_rf_regressor.pkl', 'wb') as f:
    pickle.dump(best_rf_regressor, f)
print("\nBest Random Forest Regressor model saved as 'best_rf_regressor.pkl'.")

print("\nModeling process completed.")
