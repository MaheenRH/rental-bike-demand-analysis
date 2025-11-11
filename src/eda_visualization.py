import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

def plot_hourly_trends(df):
    """Plot average hourly demand."""
    os.makedirs("figures", exist_ok=True)
    hourly_avg = df.groupby('hr')['cnt'].mean()
    plt.figure(figsize=(10,5))
    hourly_avg.plot()
    plt.title("Average Hourly Bike Rentals")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Average Count")
    plt.savefig("figures/hourly_trends.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_seasonal(df):
    """Plot seasonal rental trends."""
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(8,5))
    sns.boxplot(x='season', y='cnt', data=df, hue='season', palette='Set2', legend=False)
    plt.title("Bike Rentals by Season")
    plt.xlabel("Season")
    plt.ylabel("Count")
    plt.savefig("figures/seasonal_trends.png", dpi=300, bbox_inches='tight')
    plt.show()


def correlation_heatmap(df):
    """Show correlation among numerical features only."""
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10,6))
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap (Numeric Only)")
    plt.savefig("figures/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
