def forecast_plot(result, df):
    """Plot SARIMA forecast vs actuals."""
    import os
    os.makedirs("figures", exist_ok=True)
    
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
    plt.savefig("figures/sarima_forecast.png", dpi=300, bbox_inches='tight')
    plt.show()

