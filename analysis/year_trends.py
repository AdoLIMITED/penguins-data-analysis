import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "penguins.csv")

print(df.shape)
print(df.head())
print(df["year"].value_counts().sort_index())

# Observaciones por año y especie
counts = pd.crosstab(df["year"], df["species"])
print("\nObservaciones por año y especie:\n", counts)

# Medias por especie y año
variables = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
means = df.groupby(["species", "year"])[variables].mean().round(1)
print("\nMedias por especie y año:\n", means)

FIGURES = ROOT / "figures"
years = sorted(df["year"].unique())

# Figura 1: observaciones por año
ax = counts.plot(kind="bar", rot=0, figsize=(7, 4))
ax.set_xlabel("Year")
ax.set_ylabel("Number of penguins")
ax.set_title("Observations per year and species")
plt.tight_layout()
plt.savefig(FIGURES / "observations_per_year.png", dpi=150)
plt.close()

# Figura 2: medias por especie y año (una subgráfica por variable)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, variables):
    for species, group in means[col].groupby(level="species"):
        ax.plot(years, group.values, marker="o", label=species)
    ax.set_title(col)
    ax.set_xticks(years)
axes[0, 0].legend()
fig.suptitle("Mean measurements by species and year")
plt.tight_layout()
plt.savefig(FIGURES / "measurements_by_year.png", dpi=150)
plt.close()