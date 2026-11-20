
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "penguins.csv"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_FILE)

print("=== DATASET INFORMATION ===")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n=== FIRST FIVE ROWS ===")
print(df.head())

print("\n=== DATA TYPES ===")
print(df.dtypes)

# 2. Check missing data
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# 3. Numerical statistics
numeric_cols = df.select_dtypes(include="number").columns

print("\n=== NUMERICAL STATISTICS ===")
print(df[numeric_cols].describe())

# 4. Categorical statistics
categorical_cols = df.select_dtypes(
    include=["object", "category", "string"]
).columns

print("\n=== MOST FREQUENT CATEGORICAL VALUES ===")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))
    print("Mode:", df[col].mode().iloc[0])

# 5. Average body mass by species
print("\n=== AVERAGE BODY MASS BY SPECIES ===")
means = df.groupby("species")["body_mass_g"].mean()
print(means.round(2))

# 6. Plot species distribution
plt.figure(figsize=(8, 5))
df["species"].value_counts().plot(kind="bar")
plt.title("Number of Penguins by Species")
plt.xlabel("Species")
plt.ylabel("Number of Penguins")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "species_distribution.png", dpi=150)
plt.close()

# 7. Plot body mass distribution
plt.figure(figsize=(8, 5))
df["body_mass_g"].dropna().hist(bins=20)
plt.title("Distribution of Body Mass")
plt.xlabel("Body Mass (g)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "body_mass_histogram.png", dpi=150)
plt.close()

# 8. Plot average mass by species
plt.figure(figsize=(8, 5))
means.plot(kind="bar")
plt.title("Average Body Mass by Species")
plt.xlabel("Species")
plt.ylabel("Average Body Mass (g)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "average_mass.png", dpi=150)
plt.close()

# 9. Scatter plot
plt.figure(figsize=(8, 5))
for species, group in df.groupby("species"):
    plt.scatter(
        group["flipper_length_mm"],
        group["body_mass_g"],
        label=species,
        alpha=0.7
    )

plt.title("Body Mass vs Flipper Length")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (g)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mass_vs_flipper.png", dpi=150)
plt.close()

print("\nAnalysis finished successfully!")
print("Figures saved in:", FIGURES_DIR)
