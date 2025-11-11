from src.data_preprocessing import load_data, clean_hourly_data, map_columns
from src.eda_visualization import plot_hourly_trends, plot_seasonal, correlation_heatmap
from src.time_series_models import build_sarima, forecast_plot
from src.regression_models import prepare_data, train_and_save_models

# Load Data
day_df, hour_df = load_data('data/day.csv', 'data/hour.csv')

# Preprocess
hour_df = clean_hourly_data(hour_df)
hour_df = map_columns(hour_df)

# EDA
plot_hourly_trends(hour_df)
plot_seasonal(hour_df)
correlation_heatmap(hour_df)

# Time Series Model
sarima_result = build_sarima(day_df)
forecast_plot(sarima_result, day_df)

# Regression Models
X_train, X_test, y_train, y_test = prepare_data(hour_df)
metrics = train_and_save_models(X_train, X_test, y_train, y_test)
print(metrics)
