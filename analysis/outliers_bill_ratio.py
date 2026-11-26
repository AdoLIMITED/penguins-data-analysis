"""
Outliers and bill ratio by species.

Author: René Rojas
Dataset: Palmer Penguins (data/penguins.csv)

What it does:
1. Detects outliers with the interquartile range (IQR) method for each
   numeric variable and species.
2. Computes the bill length/depth ratio (bill_ratio) and compares it
   across species.
3. Saves two figures in the figures/ folder.

Usage (from the repository root):
    python analysis/outliers_bill_ratio.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "penguins.csv"
FIGURES = BASE / "figures"
FIGURES.mkdir(exist_ok=True)

NUMERIC_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
LABELS = {
    "bill_length_mm": "Bill length (mm)",
    "bill_depth_mm": "Bill depth (mm)",
    "flipper_length_mm": "Flipper length (mm)",
    "body_mass_g": "Body mass (g)",
}


def iqr_outlier_mask(series):
    """Boolean mask of values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)


def main():
    df = pd.read_csv(DATA)  # "NA" values are read as missing
    df = df.dropna(subset=NUMERIC_COLS)
    print(f"Rows with complete measurements: {len(df)}")

    # 1. Outliers by species and variable
    rows = []
    for species, group in df.groupby("species"):
        for col in NUMERIC_COLS:
            mask = iqr_outlier_mask(group[col])
            rows.append(
                {
                    "species": species,
                    "variable": col,
                    "n": len(group),
                    "outliers": int(mask.sum()),
                    "percentage": round(100 * mask.mean(), 1),
                }
            )
    summary = pd.DataFrame(rows)
    print("\nOutliers by species and variable (IQR method):")
    print(summary.to_string(index=False))

    # 2. Bill length/depth ratio
    df["bill_ratio"] = df["bill_length_mm"] / df["bill_depth_mm"]
    print("\nBill length/depth ratio by species:")
    print(df.groupby("species")["bill_ratio"].agg(["count", "mean", "std", "min", "max"]).round(2))

    species_list = sorted(df["species"].unique())

    # Figure 1: box plots by species (red dots are outliers)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS):
        data = [df.loc[df["species"] == s, col] for s in species_list]
        ax.boxplot(data, flierprops={"marker": "o", "markerfacecolor": "red"})
        ax.set_xticks(range(1, len(species_list) + 1))
        ax.set_xticklabels(species_list)
        ax.set_title(LABELS[col])
    fig.suptitle("Outliers by species (red dots, IQR criterion)")
    fig.tight_layout()
    fig.savefig(FIGURES / "outliers_by_species.png", dpi=150)
    plt.close(fig)

    # Figure 2: bill length/depth ratio by species
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df.loc[df["species"] == s, "bill_ratio"] for s in species_list]
    ax.boxplot(data)
    ax.set_xticks(range(1, len(species_list) + 1))
    ax.set_xticklabels(species_list)
    ax.set_ylabel("Bill length / bill depth")
    ax.set_title("Bill ratio by species")
    fig.tight_layout()
    fig.savefig(FIGURES / "bill_ratio_by_species.png", dpi=150)
    plt.close(fig)

    print("\nFigures saved in:", FIGURES)


if __name__ == "__main__":
    main()
