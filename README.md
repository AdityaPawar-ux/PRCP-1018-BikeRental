# Bike Rental Demand Prediction (PRCP-1018)

A machine learning regression project that analyzes and predicts **daily bike rental demand**
from weather and seasonal data (Capital Bikeshare, Washington D.C., 2011-2012).

## Tasks
1. Exploratory data analysis of the bike sharing dataset
2. Predict the daily rental count (`cnt`) with regression models
3. Compare multiple models and recommend the best one for production
4. Report on data challenges and the techniques used

## Dataset
- `day.csv`: 731 daily records (used for modelling)
- `hour.csv`: 17,379 hourly records (used for extra EDA)
- Source: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

## Approach
- Cleaned the data: fixed a humidity sensor error (hum = 0 on one day), removed the leaking
  columns `casual` and `registered` (cnt = casual + registered), and dropped `atemp`
  (0.99 correlated with `temp`)
- One-hot encoded season, month, weekday and weather
- Compared Linear, Ridge, Lasso, KNN, SVR, Decision Tree, Random Forest, Gradient Boosting
  and XGBoost
- Tuned the best models with GridSearchCV and 5-fold cross-validation
- Checked performance with a time-based split as well as a random split

## Results
| Model | CV R² | Test R² | RMSE | MAE |
|---|---|---|---|---|
| **Tuned XGBoost** | **0.871** | **0.889** | **666** | **453** |
| Tuned Gradient Boosting | 0.869 | 0.889 | 667 | 472 |

**Best model: Tuned XGBoost** (tuned Gradient Boosting is a close alternative). The top
drivers are temperature, year (system growth from 2011 to 2012) and season/weather.

On a time-based split (train on earlier days, test on the latest ones), scores are lower than
on the random split, so the random-split score is somewhat optimistic.

## How to run
```bash
git clone https://github.com/AdityaPawar-ux/PRCP-1018-BikeRental.git
cd PRCP-1018-BikeRental
pip install -r requirements.txt
jupyter notebook PRCP-1018-BikeRental.ipynb
```

## Citation
Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background
knowledge", Progress in Artificial Intelligence (2013): pp. 1-15.
