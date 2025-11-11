# 🚴‍♀️ Rental Bike Demand Analysis

A data science project that analyzes and forecasts rental bike demand using **Python**, **time-series forecasting (SARIMA)**, and **machine-learning regression models** such as Random Forest and Gradient Boosting.

---

## 📘 Project Overview
The goal of this project is to explore factors influencing bike rental demand and to forecast future rentals using historical data.  
It combines **Exploratory Data Analysis (EDA)**, **Feature Engineering**, and **Predictive Modeling** to help optimize resource allocation for bike-sharing systems.

---

## 🧩 Key Features
- **Data Preprocessing:** Cleaning, mapping categorical features, and handling date-time columns.  
- **Exploratory Data Analysis:** Hourly, seasonal, and correlation analysis with visual insights.  
- **Time-Series Forecasting:** SARIMA model to capture seasonality and trends in daily demand.  
- **Regression Modeling:** Random Forest and Gradient Boosting for multi-feature demand prediction.  
- **Evaluation Metrics:** R², RMSE, and MAE to compare model performance.

---

## 📊 Findings & Insights
- **Seasonality:** Demand peaks during warm months and declines sharply in winter.  
- **Weather Impact:** Temperature and humidity are strong positive predictors, while rainfall reduces demand.  
- **Temporal Patterns:** High rentals during weekday rush hours, steady leisure use on weekends.  
- **Model Performance:**  
  | Model | R² | RMSE | MAE |
  |--------|----|------|-----|
  | Random Forest | 0.624 | 110.62 | 73.02 |
  | Gradient Boosting | **0.676** | **102.74** | **70.50** |
  → Gradient Boosting achieved the highest accuracy.

---

## 🛠️ Tech Stack
`Python` • `Pandas` • `NumPy` • `Matplotlib` • `Seaborn` • `Scikit-learn` • `Statsmodels`

---

## 🚀 How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/<your-username>/rental-bike-demand-analysis.git
   cd rental-bike-demand-analysis
