#!/usr/bin/env python
from datetime import datetime, timedelta
import time
from collections import namedtuple
import requests
import glob
import os
import pandas as pd

month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
data_dir = "/home/samco/_ds/june-gloom-analysis/raw_data/"
combined_weather_file = "combined_weather_data.csv"
combined_weather_df = pd.DataFrame()


def init():
    global combined_weather_df
    combined_weather_df = pd.read_csv(data_dir + combined_weather_file)


def weather_data_features():
    global combined_weather_df, pt_june_gloom_df, fog_days_df, rain_days_df
    combined_weather_df['date'] = pd.to_datetime(combined_weather_df['date'])
    combined_weather_df['year'] = combined_weather_df['date'].apply(lambda date: date.year)
    combined_weather_df['month'] = combined_weather_df['date'].apply(lambda date: date.month)
    combined_weather_df['day'] = combined_weather_df['date'].apply(lambda date: date.day)
    combined_weather_df['dayofweek'] = combined_weather_df['date'].apply(lambda date: date.dayofweek)
    combined_weather_df.drop('Unnamed: 0', axis=1, inplace=True)
    pt_june_gloom_df = combined_weather_df.pivot_table(index='month', columns=['day', 'year'], values='fog')
    fog_days_df = combined_weather_df[(combined_weather_df['fog'] == 1) & (combined_weather_df['rain'] == 0)][
        ['month', 'year', 'meanwdird', 'meantempm', 'meandewptm', 'meanpressurem', 'fog']]
    fog_days_df.set_index('month', inplace=True)
    rain_days_df = combined_weather_df[combined_weather_df['rain'] == 1][
        ['month', 'year', 'meanwdird', 'meantempm', 'meandewptm', 'meanpressurem']]
    rain_days_df.set_index('month', inplace=True)


def wug_data_concat():
    file_1 = pd.read_csv(data_dir + "1997")
    file_2 = pd.read_csv(data_dir + "1998")
    file_3 = pd.read_csv(data_dir + "1999")
    file_4 = pd.read_csv(data_dir + "2000")
    file_5 = pd.read_csv(data_dir + "2001")
    file_6 = pd.read_csv(data_dir + "2002")
    concat_weather_df = pd.concat([file_1, file_2, file_3, file_4, file_5, file_6])
#    combined_weather_df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', data_dir + "1997")))
    concat_weather_df.to_csv(data_dir + combined_weather_file)


def wug_api_request(year):
    api_key = '8940b5f3356ab273'
    base_url = "http://api.wunderground.com/api/{}/history_{}/q/CA/ksmo.json"
    data_dir = "/home/samco/_ds/june-gloom-analysis/raw_data/"
    target_date = datetime(year, 1, 1)
    if year % 4 == 0:
        leap = 366
    else:
        leap = 365
    features = ["date", "fog", "rain", "meanwdird", "meanwindspdm", "meantempm", "meandewptm", "meanpressurem",
                "maxhumidity", "minhumidity", "maxtempm", "mintempm", "maxdewptm", "mindewptm",
                "maxpressurem", "minpressurem", "precipm"]
    DailySummary = namedtuple("DailySummary", features)

    def extract_weather_data(base_url, api_key, target_date, days):
        records = []
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
    records = extract_weather_data(base_url, api_key, target_date, leap)
    r = pd.DataFrame(records)
    r.to_csv(data_dir + '{}'.format(target_date.year), index=False)
