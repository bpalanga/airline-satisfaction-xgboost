# Airline Passenger Satisfaction: XGBoost Model

An XGBoost classifier that predicts whether an airline customer was satisfied with their flight, built from survey responses of about 130,000 customers. It continues earlier decision tree and random forest models on the same dataset.

## Dataset

`Invistico_Airline.csv`: 129,880 survey responses with 22 columns, including customer type, type of travel, class, flight distance, delays, and 1 to 5 ratings for services such as seat comfort and inflight entertainment. The target is `satisfaction` (satisfied or dissatisfied).

## Approach

1. Dropped 393 rows with missing `Arrival Delay in Minutes`.
2. One-hot encoded the categorical columns and the target.
3. Stratified 75/25 train/test split.
4. Tuned `XGBClassifier` (`binary:logistic`) with 5-fold `GridSearchCV`, refit on F1.
5. Evaluated on the held-out test set with accuracy, precision, recall, F1, and a confusion matrix.
6. Fit a decision tree and a random forest on the same split for a like-for-like comparison.

Best settings: `max_depth=8`, `min_child_weight=3`, `n_estimators=100`, `learning_rate=0.1`, `subsample=0.7`, `colsample_bytree=0.7`.

## Results (test set)

| Model | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|
| Decision tree (untuned) | 0.940 | 0.942 | 0.937 | 0.934 |
| Random forest (tuned) | 0.949 | 0.947 | 0.952 | 0.945 |
| XGBoost (tuned) | 0.958 | 0.948 | 0.968 | 0.955 |

XGBoost is best on every metric. By gain, the strongest drivers of satisfaction are inflight entertainment, seat comfort, and ease of online booking. The best `max_depth` and `n_estimators` sit at the top of the grid, so a wider search could improve the score further.

## Run it

```bash
pip install numpy pandas scikit-learn xgboost matplotlib jupyter
jupyter notebook Activity_Build_an_XGBoost_model.ipynb
```

Invistico_Airline.csv

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Jupyter.
