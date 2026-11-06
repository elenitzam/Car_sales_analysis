# Car_sales_analysis
An analysis on how Inflation and Interest Rates Impact Car Prices and Sales

## Set up and run (Windows PowerShell)

Make sure Python is installed. Open PowerShell and run:

```powershell
cd Car_sales_analysis
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python data_cleaning.py
python exploratory_analysis.py
```

Create the virtual environment and install the requirements only once. To run the project again in a new PowerShell window:

```powershell
cd Car_sales_analysis
.\.venv\Scripts\Activate.ps1
python data_cleaning.py
python exploratory_analysis.py
```


## Generated files

After you run `data_cleaning.py`, a `data` folder is created with the cleaned dataset, `auto_prices_cleaned.csv`.

After you run `exploratory_analysis.py`, a `visualizations` folder is created with all the generated plots.

## What the plots show

- `price_trends.png` — Shows new and used car prices over time. New car prices increase gradually, while used car prices fall until around mid-2022 and then rise again. This suggests that used car prices change more over time compared to new car prices.
- `inflation_vs_new_price.png` — Shows how monthly inflation rates and new car prices are related. The trend line slopes upward, suggesting a positive relationship. However new car prices vary at inflation rates between 3% and 6% so the relationship is not very consistent.
- `inflation_vs_used_price.png` — Shows the relationship between inflation rates and used car prices. The downward trend line suggests a negative relationship, with most points following the general trend. A few points at inflation rates between 0% and 2% are further from the line but not by a large amount.
- `interest_vs_units_sold.png` — Shows the relationship between monthly interest rates and units sold. The trend line slopes upward, suggesting that higher interest rates are associated with higher car sales in this dataset.
- `new_price_vs_units_sold.png` — Shows the relationship between new car prices and units sold. The upward trend line suggests a positive relationship, meaning that higher car prices are associated with higher sales in this dataset.


