# Initialize imports
import numpy as np
import pandas as pd

# Static Vars
data_dir = r'data/'
ksmo_file = 'WUG_SMO_Data_Clean.csv'


def init():
    df = pd.read_csv(data_dir + ksmo_file)
    df['mhumidity'] = (df['minhumidity'] + df['maxhumidity']) / 2
    df = df.drop(['Unnamed: 0', 'year', 'month', 'day', 'dayofweek'], axis=1)
    df = df.dropna()

    return df
