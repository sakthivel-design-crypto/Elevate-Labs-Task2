import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
df = pd.read_csv("Titanic-Dataset.csv")
print("First 5 Rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
plt.figure(figsize=(6,4))

plt.hist(df["Age"].dropna(), bins=20)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.savefig("age_histogram.png")

plt.show()
plt.figure(figsize=(6,4))

plt.hist(df["Fare"], bins=20)

plt.title("Fare Distribution")

plt.xlabel("Fare")

plt.ylabel("Frequency")

plt.savefig("fare_histogram.png")

plt.show()
plt.figure(figsize=(6,4))

sns.boxplot(x=df["Age"])

plt.title("Age Boxplot")

plt.savefig("age_boxplot.png")

plt.show()
plt.figure(figsize=(6,4))

sns.boxplot(x=df["Fare"])

plt.title("Fare Boxplot")

plt.savefig("fare_boxplot.png")

plt.show()
pair_df = df[["Survived","Pclass","Age","Fare"]]

sns.pairplot(pair_df)

plt.savefig("pairplot.png")

plt.show()
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(8,6))

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Matrix")

plt.savefig("correlation_heatmap.png")

plt.show()
print("\nObservations:")
print("1. Most passengers were between 20 and 40 years old.")
print("2. Fare is right-skewed, with a few very high values.")
print("3. Fare has several outliers.")
print("4. Passenger class appears to influence fare.")
print("5. Age contains some missing values.")
