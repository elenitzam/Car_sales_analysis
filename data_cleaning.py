"""Download, inspect, clean, and validate the auto-prices dataset.

Converted from Data_cleaning.ipynb.
Run: python data_cleaning.py
Dependencies: pandas, numpy, kagglehub
"""

# Download and display dataset
import kagglehub
import pandas as pd
from pathlib import Path

# Download dataset
dataset_path = kagglehub.dataset_download(
    "abduulwasay/auto-prices-and-economic-trends-20192023"
)

# Find all CSV files
csv_files = list(Path(dataset_path).rglob("*.csv"))

# Display available files
print("Available CSV files:")
for file in csv_files:
    print(file.name)

# Check whether CSV files were found
if not csv_files:
    raise FileNotFoundError("No CSV files found in the dataset.")

# Select the first CSV file
file_path = csv_files[0]

# Load dataset
df = pd.read_csv(file_path)

# Display first 5 records
print("\nFirst 5 records:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:", df.shape)

# Display dataset information
print("\nDataset Information:")
df.info()

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Summary statistics
print("\nNumerical Summary:")
print(df.describe())

# Categorical summary
print("\nCategorical Summary:")
print(df.describe(include=["object", "str", "category"]))

# Convert data types

# Create a copy of the original dataset
df_clean = df.copy()

# Convert date column
df_clean["Month/Year"] = pd.to_datetime(
    df_clean["Month/Year"],
    format="%y-%b",
    errors="coerce"
)

# Convert prices and units sold
for col in ["New Price ($)", "Used Price ($)"]:
    df_clean[col] = pd.to_numeric(
        df_clean[col].str.replace(",", "", regex=False),
        errors="coerce"
    )

# Convert percentage columns
for col in ["Inflation Rate (%)", "Interest Rate (%)"]:
    df_clean[col] = pd.to_numeric(
        df_clean[col].str.replace("%", "", regex=False),
        errors="coerce"
    )

df_clean["Units Sold"] = pd.to_numeric(
    df_clean["Units Sold"].str.replace(",", "", regex=False),
    errors="coerce"
).astype("Int64")



# Check updated data types
print("Updated Dataset Information:")
df_clean.info()

# Numerical summary
print("\nNumerical Summary:")
print(df_clean.describe().round(2))

# Check for missing values 
# Missing values
print("Missing Values:")
print(df_clean.isnull().sum())

# Check invalid or missing dates
invalid_dates = df_clean[df_clean["Month/Year"].isna()]

print("Invalid Dates:", len(invalid_dates))
print(invalid_dates)

df_clean = df_clean.dropna()

#  Check full-row duplicates
print("Duplicate Records:")
print(df_clean.duplicated(keep=False).sum())

# 2. Check duplicate months
duplicate_months = df_clean[
    df_clean.duplicated(
        subset=["Month/Year"],
        keep=False
    )
]

print("\nRows with Duplicate Months:", len(duplicate_months))
print(duplicate_months)

# Sort chronologically
df_clean = df_clean.sort_values(
    "Month/Year"
).reset_index(drop=True)

# Check missing months
expected_months = pd.date_range(
    start="2019-01-01",
    end="2023-12-01",
    freq="MS"
)

missing_months = expected_months.difference(
    df_clean["Month/Year"].dropna()
)

print("\nMissing Months:", missing_months.tolist())

# Define expected date range
start_date = pd.Timestamp("2019-01-01")
end_date = pd.Timestamp("2023-12-01")

# Identify dates outside the range
out_of_range = df_clean[
    (df_clean["Month/Year"] < start_date) |
    (df_clean["Month/Year"] > end_date)
]

print("Dates Outside Expected Range:", len(out_of_range))
print(out_of_range)

# Check whether dates are chronologically sorted
print(
    "Dates Chronologically Ordered:",
    df_clean["Month/Year"].is_monotonic_increasing
)
# Invalid entries check

import numpy as np
# Define validation rules
invalid_entries = pd.DataFrame(index=df_clean.index)

# 1. Invalid or missing dates
invalid_entries["Month/Year"] = (
    df_clean["Month/Year"].isna() |
    ~df_clean["Month/Year"].between(
        "2019-01-01", "2023-12-01"
    )
)

# 2. Invalid new car prices
invalid_entries["New Price ($)"] = (
    df_clean["New Price ($)"].isna() |
    (df_clean["New Price ($)"] <= 0)
)

# 3. Invalid used car prices
invalid_entries["Used Price ($)"] = (
    df_clean["Used Price ($)"].isna() |
    (df_clean["Used Price ($)"] <= 0)
)

# 4. Invalid inflation rates
# Negative inflation is valid (deflation)
invalid_entries["Inflation Rate (%)"] = (
    df_clean["Inflation Rate (%)"].isna() |
    ~np.isfinite(df_clean["Inflation Rate (%)"])
)

# 5. Invalid interest rates
# Negative rates may be economically valid
invalid_entries["Interest Rate (%)"] = (
    df_clean["Interest Rate (%)"].isna() |
    ~np.isfinite(df_clean["Interest Rate (%)"])
)

# 6. Invalid units sold
invalid_entries["Units Sold"] = (
    df_clean["Units Sold"].isna() |
    (df_clean["Units Sold"] < 0) |
    (df_clean["Units Sold"] % 1 != 0)
)

# Ensure missing comparison results are treated as invalid
invalid_entries = invalid_entries.fillna(True)

# Count invalid values per column
print("Invalid Entries per Column:")
print(invalid_entries.sum().to_frame("Invalid Count"))

# Clean column names (remove whitespace, lowercase, and replace spaces)
df_clean.columns = (
    df_clean.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.rstrip("_")
)

print("Cleaned Column Names:")
print(df_clean.columns.tolist())


#Save the cleaned dataset
output_path = Path(__file__).resolve().parent / "data" / "auto_prices_cleaned.csv"
output_path.parent.mkdir(exist_ok=True)
df_clean.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path}")