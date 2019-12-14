# Initialize imports
import pandas as pd


class Preprocess:
    def __init__(self):
        self.data_dir = r'data/'
        self.ksmo_file = 'WUG_SMO_Data_Clean.csv'
        self.klax_file = 'NOAA_LAX_Data_Clean.csv'
        self.df = pd.DataFrame()

    def ksmo(self):
        self.df = pd.read_csv(self.data_dir + self.ksmo_file)
        self.df['mhumidity'] = (self.df['minhumidity'] + self.df['maxhumidity']) / 2
        self.df = self.df.drop(['Unnamed: 0', 'year', 'month', 'day', 'dayofweek'], axis=1)
        self.df = self.df.dropna()

        return self.df

    def klax(self):
        self.df = pd.read_csv(self.data_dir + self.klax_file)
        self.df['FOG'] = self.df['FRSHTT'].apply(lambda col: 1 if col == 'fog' else 0)

        return self.df
