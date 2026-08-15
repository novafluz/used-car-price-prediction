import pandas as pd

DATAPATH = './data/used_cars.csv'
def load_data():
    return pd.read_csv(DATAPATH)