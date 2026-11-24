from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. PROJECT PATHS (compatible with repo & local Spyder)
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
BASE_DIR = SCRIPT_DIR.parent
DATA_FILE = BASE_DIR / "data" / "penguins.csv"

if DATA_FILE.exists():
    FIGURES_DIR = BASE_DIR / "figures"
else:
    # Fallback when running locally in the same folder as penguins.csv
    DATA_FILE = SCRIPT_DIR / "penguins.csv"
    FIGURES_DIR = SCRIPT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

# ==========================================
# 2. ISLAND & HABITAT ANALYSIS
# ==========================================
island_species = pd.crosstab(df["island"], df["species"])
island_species_pct = pd.crosstab(df["island"], df["species"], normalize="index") * 100

print("=== SPECIES COUNT BY ISLAND ===")
print(island_species)

print("\n=== SPECIES PERCENTAGE BY ISLAND (%) ===")
print(island_species_pct.round(1))

# Figure 1: Stacked bar chart of species distribution by island
ax = island_species.plot(kind="bar", stacked=True, figsize=(8, 5))
plt.title("Penguin Species Distribution by Island")
plt.xlabel("Island")
plt.ylabel("Number of Penguins")
plt.xticks(rotation=0)
plt.legend(title="Species")

for container in ax.containers:
    labels = [f"{int(v)}" if v > 0 else "" for v in container.datavalues]
    ax.bar_label(container, labels=labels, label_type="center", color="white", fontweight="bold")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "species_by_island.png", dpi=150)
plt.show()
plt.close()

# ==========================================
# 3. CORRELATION MATRIX OF PHYSICAL MEASUREMENTS
# ==========================================
numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
clean_labels = ["Bill Length", "Bill Depth", "Flipper Length", "Body Mass"]
corr = df[numeric_cols].corr()

print("\n=== CORRELATION MATRIX ===")
print(corr.round(3))

# Figure 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Pearson Correlation (r)")

ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(clean_labels, rotation=25, ha="right")
ax.set_yticklabels(clean_labels)
ax.set_title("Correlation Matrix of Physical Measurements")

for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        val = corr.iloc[i, j]
        text_color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
plt.show()
plt.close()

# ==========================================
# 4. BILL LENGTH VS BILL DEPTH (SIMPSON'S PARADOX)
# ==========================================
df_bill = df.dropna(subset=["bill_length_mm", "bill_depth_mm"])
overall_r = df_bill["bill_length_mm"].corr(df_bill["bill_depth_mm"])

print("\n=== BILL LENGTH VS BILL DEPTH CORRELATION (SIMPSON'S PARADOX) ===")
print(f"Overall correlation: r = {overall_r:.3f}")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = {"Adelie": "#1f77b4", "Chinstrap": "#ff7f0e", "Gentoo": "#2ca02c"}

for species, group in df_bill.groupby("species"):
    r_sp = group["bill_length_mm"].corr(group["bill_depth_mm"])
    print(f"{species} correlation: r = {r_sp:.3f}")

    ax.scatter(
        group["bill_length_mm"],
        group["bill_depth_mm"],
        label=f"{species} (r = {r_sp:+.2f})",
        color=colors[species],
        alpha=0.7,
    )

    # Trend line per species
    m, b = np.polyfit(group["bill_length_mm"], group["bill_depth_mm"], 1)
    x_line = np.linspace(group["bill_length_mm"].min(), group["bill_length_mm"].max(), 50)
    ax.plot(x_line, m * x_line + b, color=colors[species], linewidth=2)

# Overall trend line
m_all, b_all = np.polyfit(df_bill["bill_length_mm"], df_bill["bill_depth_mm"], 1)
x_all = np.linspace(df_bill["bill_length_mm"].min(), df_bill["bill_length_mm"].max(), 100)
ax.plot(
    x_all,
    m_all * x_all + b_all,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Overall trend (r = {overall_r:+.2f})",
)

ax.set_title("Bill Length vs Bill Depth by Species (Simpson's Paradox)")
ax.set_xlabel("Bill Length (mm)")
ax.set_ylabel("Bill Depth (mm)")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "bill_simpsons_paradox.png", dpi=150)
plt.show()
plt.close()

print("\nAdditional analysis finished successfully!")
print("Figures saved in:", FIGURES_DIR)