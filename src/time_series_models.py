import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

def build_sarima(df: pd.DataFrame):
    """Build SARIMA model on daily data."""
    df = df.copy()
    df['dteday'] = pd.to_datetime(df['dteday'])
    df.set_index('dteday', inplace=True)
    df = df.asfreq('D')

    train = df.iloc[:-30]
    test = df.iloc[-30:]

    model = SARIMAX(train['cnt'], order=(1,1,1), seasonal_order=(1,1,1,7))
    result = model.fit(disp=False)
    print("SARIMA model fitted.")
    return result


def forecast_plot(result, df: pd.DataFrame):
    """Plot SARIMA forecast vs actuals."""
    forecast = result.get_forecast(steps=30)
    pred = forecast.predicted_mean
    conf_int = forecast.conf_int()

    df['dteday'] = pd.to_datetime(df['dteday'])
    df.set_index('dteday', inplace=True)

    plt.figure(figsize=(10,5))
    plt.plot(df.index, df['cnt'], label='Actual')
    plt.plot(pred.index, pred, label='Forecast', color='red')
    plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink', alpha=0.3)
    plt.title("SARIMA Forecast of Bike Rentals (Next 30 Days)")
    plt.legend()
    plt.show()
