from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "penguins.csv"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MEASUREMENTS = {
    "bill_length_mm": "Bill length (mm)",
    "bill_depth_mm": "Bill depth (mm)",
    "flipper_length_mm": "Flipper length (mm)",
    "body_mass_g": "Body mass (g)",
}
SEX_COLORS = {"female": "#E69F00", "male": "#0072B2"}

df = pd.read_csv(DATA_FILE)

# 1. Summary of numeric variables (year is excluded: it is not a measurement)
numeric_summary = df[list(MEASUREMENTS)].agg(["count", "mean", "min", "max"]).T
numeric_summary["missing"] = df[list(MEASUREMENTS)].isna().sum()

print("=== NUMERIC VARIABLES ===")
print(numeric_summary.round(2))

# 2. Most frequent value of each categorical variable
print("\n=== CATEGORICAL VARIABLES ===")
for col in ["species", "island", "sex", "year"]:
    counts = df[col].value_counts()
    top = counts[counts == counts.max()].index.astype(str)
    print(
        f"{col}: most frequent = {', '.join(top)} ({counts.max()}), "
        f"missing = {df[col].isna().sum()}"
    )

# 3. Mean measurements by species and sex (rows with unknown sex are dropped)
known = df.dropna(subset=["sex", *MEASUREMENTS])
means = known.groupby(["species", "sex"])[list(MEASUREMENTS)].mean()

print(f"\nPenguins with known sex and measurements: {len(known)}")
print("\n=== MEAN BY SPECIES AND SEX ===")
print(means.round(1))

# 4. How much larger males are than females, in percent
difference = (
    means.xs("male", level="sex") / means.xs("female", level="sex") - 1
) * 100

print("\n=== MALE VS FEMALE DIFFERENCE (%) ===")
print(difference.round(1))

# 5. Boxplots by species and sex
species_list = sorted(known["species"].unique())
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (column, label) in zip(axes.flat, MEASUREMENTS.items()):
    for offset, (sex, color) in zip([-0.2, 0.2], SEX_COLORS.items()):
        data = [
            known.loc[(known["species"] == s) & (known["sex"] == sex), column]
            for s in species_list
        ]
        ax.boxplot(
            data,
            positions=[i + offset for i in range(len(species_list))],
            widths=0.35,
            patch_artist=True,
            boxprops={"facecolor": color},
            medianprops={"color": "black"},
        )
    ax.set_xticks(range(len(species_list)), species_list)
    ax.set_title(label)

fig.legend(
    handles=[Patch(color=c, label=s) for s, c in SEX_COLORS.items()],
    loc="upper right",
)
fig.suptitle("Penguin Measurements by Species and Sex")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "sex_dimorphism.png", dpi=150)
plt.close(fig)

print("\nFigure saved in:", FIGURES_DIR / "sex_dimorphism.png")
