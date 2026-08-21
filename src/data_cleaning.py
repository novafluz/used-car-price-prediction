import pandas as pd
import numpy as np

def simple_color(color):
    """Get main color from name"""
    if pd.isna(color) or color == 'Unknown':
        return 'Unknown'
    
    c = str(color).lower()
    main_colors = ['black', 'white', 'gray', 'silver', 'red', 'blue', 'brown', 'beige', 'green', 'gold']
    
    for main_color in main_colors:
        if main_color in c: 
            return main_color.capitalize()  
    
    return 'Other'

def clean_data(df):
    """Apply data cleaning to dataframe"""
    df = df.copy()

    if 'price' in df:
        # convert price and milage columns into numeric form
        df['price'] = df['price'].str.replace('[$,]', '', regex = True).astype(float)

        # remove outliers
        df['price'] = np.log1p(df['price'])

        q1 = df['price'].quantile(0.25)
        q3 = df['price'].quantile(0.75)

        iqr = q3 - q1 # interquartile range

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df = df[(df['price'] >= lower) & (df['price'] <= upper)]

    # Handle missing values
    df.fillna({'fuel_type': 'Unknown', 'accident': 'None reported', 'clean_title': 'Unknown'}, inplace = True); # semicolon to prevent printing table
    
    # Apply general data cleaning (milage, colors, accident)
    # Clean milage by removing commas and 'mi.'
    df['milage'] = df['milage'].str.replace('[, mi.]', '', regex=True).astype(float)
    
    # Group colors
    df['ext_col'] = df['ext_col'].apply(simple_color)
    df['int_col'] = df['int_col'].apply(simple_color)
    
    # Convert accident to binary
    df['accident'] = df['accident'].map({
        'At least 1 accident or damage reported': 1,
        'None reported': 0
    })
    
    return df
