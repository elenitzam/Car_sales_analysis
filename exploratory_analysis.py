"""Create exploratory visualizations for the cleaned car prices dataset.

Run:
    python exploratory_analysis.py

Dependencies:
    pandas, matplotlib, seaborn
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Find the cleaned dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "auto_prices_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "visualizations"

# Check that the cleaned dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find {DATA_PATH.name}. Run data_cleaning.py first.")

# Load the cleaned dataset
df = pd.read_csv(DATA_PATH, parse_dates=["month_year"])

if df.empty:
    raise ValueError("The cleaned dataset is empty.")

# Make sure the data is in chronological order
df = df.sort_values("month_year")

# Create the output folder if it does not already exist
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. New and used car prices over time
plt.figure(figsize=(11, 6))

plt.plot(df["month_year"], df["new_price"], label="New car price", linewidth=2)

plt.plot(df["month_year"], df["used_price"], label="Used car price", linewidth=2)

plt.title("New and Used Car Prices Over Time")
plt.xlabel("Month")
plt.ylabel("Price ($)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, color="lightgray", alpha=0.7)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "price_trends.png", dpi=300)
plt.close()


# 2. Inflation rate and new car prices
plt.figure(figsize=(8, 6))

sns.regplot(data=df, x="inflation_rate", y="new_price", ci=None, scatter_kws={"alpha": 0.75})

plt.title("Inflation Rate vs. New Car Prices")
plt.xlabel("Inflation Rate (%)")
plt.ylabel("New Car Price ($)")
plt.grid(True, color="lightgray", alpha=0.7)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "inflation_vs_new_price.png", dpi=300)
plt.close()

# 3. Inflation rate and used car prices
plt.figure(figsize=(8, 6))

sns.regplot(data=df, x="inflation_rate", y="used_price", ci=None, scatter_kws={"alpha": 0.75})

plt.title("Inflation Rate vs. Used Car Prices")
plt.xlabel("Inflation Rate (%)")
plt.ylabel("Used Car Price ($)")
plt.grid(True, color="lightgray", alpha=0.7)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "inflation_vs_used_price.png", dpi=300)
plt.close()

# 4. Interest rates and units sold
plt.figure(figsize=(8, 6))

sns.regplot(data=df, x="interest_rate", y="units_sold", ci=None, scatter_kws={"alpha": 0.75})

plt.title("Interest Rate vs. Units Sold")
plt.xlabel("Interest Rate (%)")
plt.ylabel("Units Sold")
plt.grid(True, color="lightgray", alpha=0.7)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "interest_vs_units_sold.png", dpi=300)
plt.close()

# 5. New car prices and units sold
plt.figure(figsize=(8, 6))

sns.regplot(data=df, x="new_price", y="units_sold", ci=None, scatter_kws={"alpha": 0.75})

plt.title("New Car Prices vs. Units Sold")
plt.xlabel("New Car Price ($)")
plt.ylabel("Units Sold")
plt.grid(True, color="lightgray", alpha=0.7)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "new_price_vs_units_sold.png", dpi=300)
plt.close()


print("Exploratory analysis completed successfully.")
print(f"Visualizations saved in: {OUTPUT_DIR}")
print("Created charts:")
for image_path in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"- {image_path.name}")
