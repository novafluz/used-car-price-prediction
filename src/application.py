from train import train_models
from loadData import load_data
from preprocessing import preprocess_data
import numpy as np
import pandas as pd

def app():
    print('=' * 65)
    print('Used Car Price Prediction')
    print('=' * 65)
    models, preprocessor, _, _ = train_models()
    xgboost_model = models['XGBoost']
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
    
    X_user = preprocess_data(user_df, preprocessor)
    
    predicted_price = np.expm1(xgboost_model.predict(X_user))
    
    print(f"Predicted Car Price: ${predicted_price[0]:.2f}")