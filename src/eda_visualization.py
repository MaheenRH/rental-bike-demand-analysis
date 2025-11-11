import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

def plot_hourly_trends(df: pd.DataFrame):
    """Plot average hourly demand."""
    hourly_avg = df.groupby('hr')['cnt'].mean()
    plt.figure(figsize=(10,5))
    hourly_avg.plot()
    plt.title("Average Hourly Bike Rentals")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Average Count")
    plt.show()


def plot_seasonal(df: pd.DataFrame):
    """Plot seasonal rental trends."""
    plt.figure(figsize=(8,5))
    sns.boxplot(x='season', y='cnt', data=df, palette='Set2')
    plt.title("Bike Rentals by Season")
    plt.xlabel("Season")
    plt.ylabel("Count")
    plt.show()


def correlation_heatmap(df: pd.DataFrame):
    """Show correlation among numerical features only."""
    plt.figure(figsize=(10,6))
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap (Numeric Features Only)")
    plt.show()
