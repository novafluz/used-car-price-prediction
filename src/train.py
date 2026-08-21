from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from loadData import load_data
from preprocessing import preprocess_data

def train_models():
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(load_data())

    # ========= Train Linear Regression =========
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    # ========= Train Random Forest =========
    random_forest_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=67
        )
    random_forest_model.fit(X_train, y_train)

    # ========= Train XGBoost =========
    xgboost_model = XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate = 0.1,
        random_state=67
        )
    xgboost_model.fit(X_train, y_train)

    return {
        'Linear Regression': regression_model,
        'Random Forest': random_forest_model,
        'XGBoost': xgboost_model
    }, preprocessor, X_test, y_test


if __name__ == '__main__':
    train_models()
