from train import regression_model, random_forest_model, xgboost_model
from preprocessing import X_test, y_test
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

models = {
    'Linear Regression': regression_model,
    'Random Forest': random_forest_model,
    'XGBoost': xgboost_model
}
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"Test set's R2 score loss for {name}: {r2_score(y_true=y_test, y_pred=y_pred)}")
    print(f"Test set's RMSE score loss for {name}: {root_mean_squared_error(y_true=y_test, y_pred=y_pred)}")
    print(f"Test set's MAE score loss for {name}: {mean_absolute_error(y_true=y_test, y_pred=y_pred)}")
    print()