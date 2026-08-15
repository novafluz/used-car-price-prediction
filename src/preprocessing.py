import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from loadData import load_data
from feature_engineering import apply_features
from data_cleaning import clean_data

car_data = load_data()

# =========== Data Cleaning ===========
# convert price and milage columns into numeric form
car_data['price'] = car_data['price'].str.replace('[$,]', '', regex = True).astype(float)

# remove outliers
car_data = car_data[(car_data['price'] > 500) & (car_data['price'] < 500_000)]

# Handle missing values
car_data.fillna({'fuel_type': 'Unknown', 'accident': 'None reported', 'clean_title': 'Unknown'}, inplace = True); # semicolon to prevent printing table

# Apply general data cleaning (milage, colors, accident)
car_data = clean_data(car_data)

# =========== Feature Engineering ===========
car_data = apply_features(car_data)

# =========== Split ===========
X = car_data.drop(columns = ['price'])
y = car_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=67)

# =========== Preprocess using Scaler and OneHot Encoder ===========

categorical_cols = ['brand', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'clean_title']
numeric_cols = ['milage', 'car_age', 'accident', 'milage_per_year', 'engine_hp', 'engine_size', 'engine_cylinders']

# show correlation of features
all_numeric_cols = numeric_cols + ['price']
FIG_PATH = './fig'
corr = car_data[all_numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cbar=True, cmap='Blues')
plt.title("Correlation of features")
plt.savefig(f'{FIG_PATH}/correlation')
plt.close()

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

if __name__ == "__main__":
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")