# ============================================
# Exploratory Data Analysis (EDA) on Titanic Dataset
# ============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create output folder
os.makedirs("output", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/titanic.csv")

# ============================================
# Dataset Overview
# ============================================

print("="*50)
print("First 5 Rows")
print("="*50)
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================
# Statistical Summary
# ============================================

print("\nStatistical Summary")
print(df.describe())

print("\nComplete Summary")
print(df.describe(include='all'))

# ============================================
# Fill Missing Values
# ============================================

df["Age"] = df["Age"].fillna(df["Age"].median())

if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# ============================================
# Survival Count
# ============================================

plt.figure(figsize=(6,4))
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Passengers")
plt.tight_layout()
plt.savefig("output/survival_count.png")
plt.show()

# ============================================
# Gender Distribution
# ============================================

plt.figure(figsize=(6,4))
df["Sex"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("output/gender_distribution.png")
plt.show()

# ============================================
# Passenger Class Distribution
# ============================================

plt.figure(figsize=(6,4))
df["Pclass"].value_counts().sort_index().plot(kind="bar")
plt.title("Passenger Class Distribution")
plt.xlabel("Class")
plt.ylabel("Passengers")
plt.tight_layout()
plt.savefig("output/passenger_class.png")
plt.show()

# ============================================
# Age Distribution
# ============================================

plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/age_distribution.png")
plt.show()

# ============================================
# Fare Distribution
# ============================================

plt.figure(figsize=(8,5))
plt.hist(df["Fare"], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/fare_distribution.png")
plt.show()

# ============================================
# Correlation Matrix
# ============================================

numeric_df = df.select_dtypes(include=np.number)

corr = numeric_df.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png")
plt.show()

# ============================================
# Top Correlations with Survival
# ============================================

print("\nCorrelation with Survival")
print(corr["Survived"].sort_values(ascending=False))

# ============================================
# Final Insights
# ============================================

print("\n" + "="*50)
print("EDA COMPLETED SUCCESSFULLY")
print("="*50)

print("""
Key Insights:
1. Dataset explored successfully.
2. Missing values handled.
3. Statistical summary generated.
4. Survival distribution visualized.
5. Gender distribution analyzed.
6. Passenger class analyzed.
7. Age distribution analyzed.
8. Fare distribution analyzed.
9. Correlation heatmap generated.
10. Key factors affecting survival identified.

All charts are saved in the 'output' folder.
""")