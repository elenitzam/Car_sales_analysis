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
- `actual vs predicted_2023.png` - Shows that kNN and Decision tree regression models inderst underestimated car sales and failed to capture the increasing sales trend in 2023. This suggests that the models struggled to adapt to the changes in the car market during the testing period.

## Modeling analysis

The data was sorted by month, with the final 12 months (2023) held out for testing. Each model used new and used car prices, inflation, interest rates and 

### ??a month index (that we added to give the models a way to capture the overall trend in car sales over time) to predict units sold.

Linear Regression — Fits a linear relationship between the input features and car sales.

Polynomial Regression — Adds degree 2 feature combinations before fitting a linear regression model, allowing it to represent more curved relationships.

k-NN Regression — The model uses StandardScaler to standardize the predictor variables before predicting car sales based on the nearest training observations. GridSearchCV was used to test values of k from 1 to 8 using five-fold time-series cross-validation, preserving the chronological order of the data. The optimal k was selected based on the lowest average cross-validation Mean Absolute Error (MAE), and the optimized model was then evaluated on the 2023 test data.

Decision Tree Regression — The model predicts car sales by dividing the data into smaller groups based on the predictor variables. GridSearchCV was used to evaluate different maximum tree depths (2, 3, 4, 5, and unlimited) to identify the best-performing configuration. A maximum depth of 3 was selected based on the lowest cross-validation error.


## Regression Model Results

| Model | Test MAE | Test RMSE | Test R² |
| kNN Regression | 11591.67 | 12598.38 | -5.5176 |
| Decision Tree Regression | 24525.00 | 28248.05 | -31.7669 |


The models were compared using MAE, RMSE and R² on the 2023 test period. 
Linear Regression achieved the best performance, with a MAE of 3109.57, an RMSE of 4363.63 and an R² of 0.22. The R² value of 0.22 indicated that the specific model explained approximately 22% of the variation in car sales during 2023, while the remaining 78% was not explained by the model.
The Polynomial Regression also produced poor results???
The k-NN with a MAE of 11591.67, an RMSE of 12598.38 and an R² of -5.5176 and the Decision Tree model with a MAE of 24525.00, an RMSE of 28248.05 and an R² of -31.7669, performed significantly worse. The results suggest that the models struggled to predict future cars sales accurately. The reasons behind it might be the limited dataset or the fact tha new car prices increased significantly in 2023 comparing to the training period 2019-2022.
Overall, the Linear Regression was the most suitable of the four models for this dataset, while more complex models did not improve the predictions. However, even thought the Linear Regression produced the lowest prediction erros, the model still has limited predictive accuracy, suggesting that other factors not included in this analysis, may also influence car sales.