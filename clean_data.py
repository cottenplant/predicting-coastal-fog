import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['text.color'] = 'k'
plt.rcParams['figure.figsize'] = 12, 8


def init():
    # Dirs
    data_dir = r'data/'

    # File path
    wug_data_file = 'WUG_SMO_Data_Clean.csv'
    noaa_data_file = 'NOAA_LAX_Data_Clean.csv'

    # Clean up data
    df_smo = clean_data_wug(data_dir, wug_data_file)
    df_lax = clean_data_noaa(data_dir, noaa_data_file)

    return df_smo


def clean_data_wug(directory, filename):
    df = pd.read_csv(directory + filename)
    df = df.drop(['date', 'maxhumidity', 'minhumidity', 'maxtempm', 'mintempm', 'maxdewptm', 'mindewptm',
                  'maxpressurem', 'minpressurem', 'year', 'month', 'day', 'dayofweek', 'Unnamed: 0'], axis=1)
    df.columns = ['fog', 'rain', 'mdir', 'mspd', 'mtmp', 'mdew', 'mpressure', 'precipm']
    df = df.dropna()
    return df


def clean_data_noaa(directory, filename):
    df = pd.read_csv(directory + filename)
    df = df.set_index('YEARMODA')
    return df
