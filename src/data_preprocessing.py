import pandas as pd

def load_data(day_path: str, hour_path: str):
    """Load daily and hourly bike rental datasets."""
    day_df = pd.read_csv(day_path)
    hour_df = pd.read_csv(hour_path)
    print(f"Day data shape: {day_df.shape}, Hour data shape: {hour_df.shape}")
    return day_df, hour_df


def clean_hourly_data(df: pd.DataFrame):
    """Basic cleaning — remove nulls, fix datatypes."""
    df = df.dropna()
    df['dteday'] = pd.to_datetime(df['dteday'])
    df = df.sort_values('dteday')
    return df


def map_columns(df: pd.DataFrame):
    """Map encoded columns to human-readable labels."""
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    weather_map = {
        1: 'Clear', 
        2: 'Mist', 
        3: 'Light Snow/Rain', 
        4: 'Heavy Rain/Ice'
    }
    df['season'] = df['season'].map(season_map)
    df['weathersit'] = df['weathersit'].map(weather_map)
    return df
