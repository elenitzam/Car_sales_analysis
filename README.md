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
python regression_models.py
```

Create the virtual environment and install the requirements only once. To run the project again in a new PowerShell window:

```powershell
cd Car_sales_analysis
.\.venv\Scripts\Activate.ps1
python data_cleaning.py
python exploratory_analysis.py
python regression_models.py
```


## Generated files

After you run `data_cleaning.py`, a `data` folder is created with the cleaned dataset, `auto_prices_cleaned.csv`.

After you run `exploratory_analysis.py`, a `visualizations` folder is created with all the generated plots.

After you run `regression_models.py`, it saves `actual_vs_predicted_2023.png` in the `visualizations` folder. The plot compares actual 2023 car sales with predictions from all four regression models.


## Data Exploration and Cleaning

`data_cleaning.py` downloads the dataset from KaggleHub and inspects its rows, columns, data types, and summary statistics. It uses the CSV file found in the downloaded dataset, so running this step requires an internet connection.

The script then:

- Converts the month/year field to a date.
- Removes commas from price and sales values and percent signs from inflation and interest rates, then converts these fields to numeric values.
- Prints missing value and invalid date information, then drops every row containing a missing value.
- Checks and reports duplicate rows and months, missing months, dates outside 2019–2023 and values that fail its validation rules (such as non-positive prices or negative sales).
- Sorts the remaining rows chronologically and standardizes column names.

The cleaned dataset is saved as `data/auto_prices_cleaned.csv`. 


## What the plots show

- `price_trends.png` — Shows new and used car prices over time. New car prices increase gradually, while used car prices fall until around mid-2022 and then rise again. This suggests that used car prices change more over time compared to new car prices.
- `inflation_vs_new_price.png` — Shows how monthly inflation rates and new car prices are related. The trend line slopes upward, suggesting a positive relationship. However new car prices vary at inflation rates between 3% and 6% so the relationship is not very consistent.
- `inflation_vs_used_price.png` — Shows the relationship between inflation rates and used car prices. The downward trend line suggests a negative relationship, with most points following the general trend. A few points at inflation rates between 0% and 2% are further from the line but not by a large amount.
- `interest_vs_units_sold.png` — Shows the relationship between monthly interest rates and units sold. The trend line slopes upward, suggesting that higher interest rates are associated with higher car sales in this dataset.
- `new_price_vs_units_sold.png` — Shows the relationship between new car prices and units sold. The upward trend line suggests a positive relationship, meaning that higher car prices are associated with higher sales in this dataset.
- `actual_vs_predicted_2023.png` — Compares actual 2023 car sales with predictions from all four regression models.

## Modeling analysis

The data was sorted by month, with the final 12 months (2023) held out for testing. Each model used new and used car prices, inflation, interest rates and a month index (that we added to give the models a way to capture the overall trend in car sales over time) to predict units sold.

- **Linear Regression** — Fits a linear relationship between the input features and car sales.
- **Polynomial Regression** — Adds degree 2 feature combinations before fitting a linear regression model, allowing it to represent more curved relationships.
- **k-NN Regression** — The model uses StandardScaler to standardize the predictor variables before predicting car sales based on the nearest training observations. GridSearchCV was used to test values of k from 1 to 8 using five-fold time-series cross-validation, preserving the chronological order of the data. The optimal k was selected based on the lowest average cross-validation Mean Absolute Error (MAE), and the optimized model was then evaluated on the 2023 test data.
- **Decision Tree Regression** — The model predicts car sales by dividing the data into smaller groups based on the predictor variables. GridSearchCV was used to evaluate different maximum tree depths (2, 3, 4, 5, and unlimited) to identify the best-performing configuration. A maximum depth of 3 was selected based on the lowest cross-validation error.

## Regression Model Results

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Linear Regression | 3109.567 | 4363.631 | 0.218 |
| Polynomial Regression | 20626.189 | 23342.362 | -21.374 |
| kNN Regression | 10791.667 | 11866.444 | -4.782 |
| Decision Tree Regression | 24525.00 | 28248.053 | -31.767 |

The models were compared using MAE, RMSE and R² on the 2023 test data.

Linear Regression achieved the best performance, with an MAE of 3109.57, an RMSE of 4363.63 and an R² of 0.218. This means that the model explained approximately 21.8% of the variation in car sales during 2023.

Polynomial Regression produced much worse results, with an R² of -21.374. Adding more complex feature combinations did not improve its predictions.

The k-NN model achieved an MAE of 10791.67, an RMSE of 11866.44 and an R² of -4.782. The Decision Tree performed worst, with an MAE of 24525.00, an RMSE of 28248.05 and an R² of -31.767. The results suggest that the models struggled to predict future cars sales accurately. The reasons behind it might be the limited dataset or the fact tha new car prices increased significantly in 2023 comparing to the training period 2019-2022.

The results show that we need to be careful about how complex a model is. Linear Regression is simpler and may not capture every pattern in the data. More complex models may fit the training data too closely and perform worse on new data. In our case, Linear Regression performed better on the test data than the other models.

## Interpretation and Key Findings

The main goal of this analysis was to understand how inflation and interest rates impact car prices and sales.

### 1. Do increases in interest rates directly reduce car sales?

Interest rates are often expected to affect car sales because higher rates can make car loans more expensive making it harder for customers to afford a vehicle. However, our analysis shows a positive relationship between interest rates and car sales, with the trend line in the visualization sloping upward. This means that higher interest rates were associated with higher sales in our dataset. Therefore, we cannot conclude that increasing interest rates directly reduced car sales. Other factors may also have influenced sales.

### 2. Do used car prices react faster to economic conditions than new car prices?

Our analysis shows that used and new car prices followed different patterns over time. New car prices increased gradually while used car prices decreased until around mid 2022 and then started rising again. The relationship plots show a positive relationship between inflation and new car prices but a negative relationship between inflation and used car prices, so the two markets respond differently to economic conditions. Based on our analysis we can see that used car prices changed more over time but we cannot say for sure that they react faster to economic changes

### 3. Which forecasting model performed best?

Linear Regression achieved the best results among the four models, with an R² of 0.218. The more complex models did not improve the predictions. However, even though Linear Regression produced the lowest prediction errors, its predictive accuracy was still limited. This suggests that other factors not included in this analysis may also influence car sales.