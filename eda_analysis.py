import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic.csv")

print("Dataset Info")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("screenshots/survival_count.png")
plt.show()

# Passenger Class Distribution
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.savefig("screenshots/passenger_class.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("screenshots/correlation_heatmap.png")
plt.show()

print("\nKey Insights:")
print("- First class passengers had higher survival rates.")
print("- Female passengers survived more often.")
print("- Fare has a positive relation with survival.")