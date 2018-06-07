#!/usr/bin/env python
import pandas as pd

# Static Vars
data_dir = r"raw_data/"
daily_noaa_file = "CDO4274757652424.txt"
enso_sst_file = "sst_anomaly_3month_mean.csv"
dict_3mo_mean = {1: 'DJF', 2: 'JFM', 3: 'FMA', 4: 'MAM', 5: 'AMJ', 6: 'MJJ',
                 7: 'JJA', 8: 'JAS', 9: 'ASO', 10: 'SON', 11: 'OND', 12: 'NDJ'}
dict_enso_sst = {}
df_noaa = pd.DataFrame()
df_enso_sst = pd.DataFrame()


def init():
    global df_noaa, df_enso_sst
    df_noaa = pd.read_csv(data_dir + daily_noaa_file)
    df_enso_sst = pd.read_csv(data_dir + enso_sst_file)


def noaa_clean():
    global df_noaa, df_lax
    df_lax = df_noaa.loc[df_noaa['STN---'] == 722950].copy()
    df_lax.reset_index(inplace=True)
    df_lax.drop(['index', 'STN---', 'WBAN ', '  ', '  .1', '  .2', '  .3', '  .4', '  .5', 'Unnamed: 22'], axis=1,
                inplace=True)
    df_lax.columns = df_lax.columns.str.strip()
    df_lax['YEARMODA'] = pd.to_datetime(df_lax['YEARMODA'], format='%Y%m%d')
    df_lax['month'] = df_lax['YEARMODA'].apply(lambda date: date.month)
    df_lax['year'] = df_lax['YEARMODA'].apply(lambda date: date.year)
    df_lax['MAX'] = df_lax['MAX'].apply(lambda col: float(col.strip('*')))
    df_lax['MIN'] = df_lax['MIN'].apply(lambda col: float(col.strip('*')))
    df_lax['PRCP'] = df_lax['PRCP'].apply(lambda col: float(col[:-1].strip()))
    df_lax['PRCP'] = df_lax['PRCP'].apply(impute_null_prcp)
    df_lax['dummy'] = 1


def impute_null_prcp(col):
    if col == 99.99:
        return 0.0
    else:
        return col


def feature_sst_anomaly():
    global df_lax, df_enso_sst, dict_enso_sst
    df_enso_sst.set_index('Year', inplace=True)
    dict_enso_sst = df_enso_sst.to_dict()
    df_lax['3mo_mean'] = df_lax['month'].map(dict_3mo_mean)
    df_lax['sst_anomaly'] = df_lax[['3mo_mean', 'year']].apply(sst_anomaly_lookup, axis=1)
    df_lax.drop('3mo_mean', axis=1, inplace=True)


def sst_anomaly_lookup(cols):
    mean = cols[0]
    year = cols[1]
    return dict_enso_sst[mean][year]
