# machine-learning-project


India has the largest pollution-related death toll in the world (Hayward, 2021).
We hypothesized that an increase of factors such as AQI, PM2.5, PM10, NO2, NH3, SO2, CO, ozone, temperature and humidity will lead to higher hospital admissions due to respiratory diseases in Punjab, India.
We also considered demographic factors of mortalities in Ludhiana, the subregion where our dataset hospital is located.

## 📊 Research Summary

This project investigates the relationship between air quality metrics and hospital admission durations in Punjab, India.  
We trained several regression models (Linear Regression, Random Forest, Tuned RF) to predict hospital stay duration using AQI, PM2.5, PM10, and demographic data.

Key findings:
- **Age at admission** was the strongest predictor of hospital stay duration (importance score: 0.5).
- **Air quality metrics** like AQI, PM2.5, and PM10 were highly correlated, but had limited predictive power individually.
- The **best model (Random Forest)** achieved an R² of 0.33, highlighting limitations in available features.
- Results emphasize the need for richer datasets including medical history and hospital capacity.

<p align="center">
  <img src="Screenshots/feature_importance.png" width="400" alt="Feature Importance"/>
</p>
