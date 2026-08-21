import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from feature_engineering import apply_features
from data_cleaning import clean_data

categorical_cols = ['brand', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'clean_title']
numeric_cols = ['milage', 'car_age', 'accident', 'milage_per_year', 'engine_hp', 'engine_size', 'engine_cylinders']

def preprocess_data(car_data, preprocessor=None):
    # =========== Data Cleaning ===========
    car_data = clean_data(car_data)

    # =========== Feature Engineering ===========
    car_data = apply_features(car_data)

    if preprocessor is not None:
        return preprocessor.transform(car_data.drop(columns = ['price'], errors='ignore'))

    # =========== Split ===========
    X = car_data.drop(columns = ['price'])
    y = car_data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=67)

    # =========== Preprocess using Scaler and OneHot Encoder ===========
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            # handle_unknown is ignored to avoid errors when test data doesn't contain classes in training data
            # sparse = False -> encoded cols are returned as np array instead of sparse matrix
        ]
    )

    # only fit the model to training data and apply to both train and test data to avoid data leakage
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor