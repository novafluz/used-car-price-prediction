from datetime import datetime

def apply_features(df):
    """Apply feature engineering to dataframe"""
    df = df.copy()
    
    # create feature car_age
    current_year = datetime.now().year
    df['car_age'] = current_year - df['model_year']
    
    # create miles per year
    df['milage_per_year'] = df['milage'] / df['car_age']
    
    # Extract HP, volume, cylinder
    df['engine_hp'] = df['engine'].str.extract(r'(\d+\.?\d*)\s*HP').astype(float)
    df['engine_size'] = df['engine'].str.extract(r'(\d+\.?\d*)\s*L').astype(float)
    df['engine_cylinders'] = df['engine'].str.extract(r'(\d+)\s*Cylinder').astype(float)
    
    # Handle missing values
    df['engine_hp'] = df['engine_hp'].fillna(df['engine_hp'].median())
    df['engine_size'] = df['engine_size'].fillna(df['engine_size'].median())
    df['engine_cylinders'] = df['engine_cylinders'].fillna(df['engine_cylinders'].median())
    
    return df
