from train import xgboost_model
from preprocessing import preprocessor, numeric_cols, categorical_cols
from loadData import load_data
from feature_engineering import apply_features
from data_cleaning import clean_data
import pandas as pd

def app():
    print('=' * 65)
    print('Used Car Price Prediction')
    print('=' * 65)
    car_data = load_data()
    cols = car_data.columns.drop('price')
    sample_input = ['Toyota', 
                    'Camry',
                    '2020',
                    '50000', 
                    'Gasoline', 
                    '2.5L 4 Cylinder', 
                    'Automatic', 
                    'White',
                    'Black',
                    'None reported/At least 1 accident or damage reported',
                    'Yes'
                    ]
    user_input = []
    for id, col in enumerate(cols):
        user_input.append(input(f"Enter {col} (eg. {sample_input[id]}): "))
        
    user_df = pd.DataFrame([user_input], columns=cols) # convert to df for preprocessing and prediction
    
    user_df['model_year'] = user_df['model_year'].astype(int)
    
    # Apply data cleaning
    user_df = clean_data(user_df)
    
    # Apply feature engineering
    user_df = apply_features(user_df)
    
    X_user = preprocessor.transform(user_df)
    predicted_price = xgboost_model.predict(X_user)
    
    print(f"Predicted Car Price: ${predicted_price[0]:.2f}")