import pandas as pd

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
