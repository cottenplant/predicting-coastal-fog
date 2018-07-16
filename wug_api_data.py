#!/usr/bin/env python
from datetime import datetime, timedelta
import glob
import requests
import time
from collections import namedtuple
import pandas as pd

# Static Vars
data_dir = r"raw_data/"
wug_temp_dir = r"raw_data/wug_api_import"
combined_weather_file = "combined_weather_data.csv"
enso_sst_file = "sst_anomaly_3month_mean.csv"
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
combined_weather_df = pd.DataFrame()
df_enso_sst = pd.DataFrame()
api_key = '8940b5f3356ab273'
base_url = "http://api.wunderground.com/api/{}/history_{}/q/CA/ksmo.json"
records = []


def init():
    global combined_weather_df, df_enso_sst
    combined_weather_df = pd.read_csv(data_dir + combined_weather_file)
    df_enso_sst = pd.read_csv(data_dir + enso_sst_file)


def wug_data_clean():
    global combined_weather_df
    combined_weather_df['date'] = pd.to_datetime(combined_weather_df['date'])
    combined_weather_df['year'] = combined_weather_df['date'].apply(lambda date: date.year)
    combined_weather_df['month'] = combined_weather_df['date'].apply(lambda date: date.month)
    combined_weather_df['day'] = combined_weather_df['date'].apply(lambda date: date.day)
    combined_weather_df['dayofweek'] = combined_weather_df['date'].apply(lambda date: date.dayofweek)
    combined_weather_df.drop('Unnamed: 0', axis=1, inplace=True)
    combined_weather_df['precipm'] = combined_weather_df['precipm'].apply(impute_trace_precip)
    combined_weather_df['precipm'] = combined_weather_df['precipm'].apply(lambda col: float(col))


def wug_data_features():
    global combined_weather_df, pt_june_gloom_df, fog_days_df, rain_days_df, pt_temp_df, min_max_temp_df, \
        pt_fog_by_month, year_min, year_max
    year_min = combined_weather_df['year'].unique().min()
    year_max = combined_weather_df['year'].unique().max()
    pt_june_gloom_df = combined_weather_df.pivot_table(index='month', columns=['day', 'year'], values='fog')
    fog_days_df = combined_weather_df[(combined_weather_df['fog'] == 1) & (combined_weather_df['rain'] == 0)][
        ['month', 'year', 'meanwdird', 'meantempm', 'meandewptm', 'meanpressurem', 'fog']]
    fog_days_df.set_index('month', inplace=True)
    rain_days_df = combined_weather_df[combined_weather_df['rain'] == 1][['month', 'year', 'meanwdird', 'meantempm',
                                                                          'meandewptm', 'meanpressurem']]
    rain_days_df.set_index('month', inplace=True)
    pt_temp_df = combined_weather_df.pivot_table(index='month', columns=['day', 'year'],
                                                 values=['meantempm'])
    min_max_temp_df = combined_weather_df[['date', 'year', 'mintempm', 'maxtempm']].copy()
    min_max_temp_df['month day'] = min_max_temp_df['date'].apply(lambda date: (date.month, date.day))
    fog_days_df.reset_index(inplace=True)
    pt_fog_by_month = fog_days_df.pivot_table(values='fog', index='month', columns=['year'], aggfunc='sum',
                                              fill_value=0)
    pt_fog_by_month = pt_fog_by_month.unstack(level='year')


def impute_trace_precip(col):
    if col == 'T':
        return 0.25
    else:
        return col


def wug_data_concat():
    global data_dir, wug_temp_dir
    concat_weather_df = pd.concat(map(pd.read_csv, glob.glob(wug_temp_dir + '/*')))
    concat_weather_df.to_csv(data_dir + combined_weather_file)


def wug_api_request(year):
    global wug_temp_dir, target_date
    target_date = datetime(year, 1, 1)
    if year % 4 == 0:
        leap = 366
    else:
        leap = 365
    features = ["date", "fog", "rain", "meanwdird", "meanwindspdm", "meantempm", "meandewptm", "meanpressurem",
                "maxhumidity", "minhumidity", "maxtempm", "mintempm", "maxdewptm", "mindewptm",
                "maxpressurem", "minpressurem", "precipm"]
    DailySummary = namedtuple("DailySummary", features)

    def extract_weather_data(days):
        global records, api_key, base_url, target_date
        for _ in range(days):
            request = base_url.format(api_key, target_date.strftime('%Y%m%d'))
            response = requests.get(request)
            if response.status_code == 200:
                data = response.json()['history']['dailysummary'][0]
                records.append(DailySummary(
                    date=target_date,
                    fog=data['fog'],
                    rain=data['rain'],
                    meanwdird=data['meanwdird'],
                    meanwindspdm=data['meanwindspdm'],
                    meantempm=data['meantempm'],
                    meandewptm=data['meandewptm'],
                    meanpressurem=data['meanpressurem'],
                    maxhumidity=data['maxhumidity'],
                    minhumidity=data['minhumidity'],
                    maxtempm=data['maxtempm'],
                    mintempm=data['mintempm'],
                    maxdewptm=data['maxdewptm'],
                    mindewptm=data['mindewptm'],
                    maxpressurem=data['maxpressurem'],
                    minpressurem=data['minpressurem'],
                    precipm=data['precipm']))
            time.sleep(6)
            target_date += timedelta(days=1)
        return records
    records = extract_weather_data(leap)
    r = pd.DataFrame(records)
    r.to_csv(wug_temp_dir + '{}'.format(target_date.year), index=False)
