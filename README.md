# prediction-intervals-xgboost

출처1 : https://colab.research.google.com/drive/1KlRkrLi7JmVpprL94vN96lZU-HyFNkTq#scrollTo=y7LIgbTUz2RR

출처2 : https://towardsdatascience.com/regression-prediction-intervals-with-xgboost-428e0a018b


```
#XGBoost hyper-parameter tuning
def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5, 7, 10, 13],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.4,0.5,0.6,0.7],
        'colsample_bytree': [0.4,0.5,0.6,0.7],
        'n_estimators' : [100, 200, 300],
        'min_samples_split' : [7,9,11],
        'objective': ['reg:squarederror']
    }

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_
```
