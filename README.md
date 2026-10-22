# Bike Rental Demand Prediction (PRCP-1018)

Machine learning project that analyzes and predicts daily bike rental demand
from weather and seasonal data (Capital Bikeshare, Washington D.C., 2011-2012).

## Tasks
1. Exploratory data analysis of the bike sharing dataset
2. Predict daily rental count (`cnt`) with regression models
3. Compare multiple models and recommend the best for production
4. Report on data challenges and the techniques used

## Dataset
- `day.csv`: 731 daily records (used for modelling)
- `hour.csv`: 17,379 hourly records (used for extra EDA)
- Source: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

## Approach
- Cleaned data (fixed a humidity sensor error, handled leakage and multicollinearity)
- Compared Linear, Ridge, Lasso, KNN, SVR, Decision Tree, Random Forest,
  Gradient Boosting and XGBoost
- Tuned the best models with GridSearchCV and 5-fold cross-validation

## Results
- Best model: (write your model name here)
- CV R²: (your number) | Test R²: (your number) | RMSE: (your number) | MAE: (your number)

## How to run
pip install -r requirements.txt
jupyter notebook PRCP-1018-BikeRental.ipynb

## Citation
Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors
and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15.
