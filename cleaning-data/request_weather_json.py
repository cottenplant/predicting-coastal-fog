import json

api_key = '8940b5f3356ab273'
base_url = "http://api.wunderground.com/api/{}/conditions/q/{}.json"

json_file = "api_result.json"

records = []

with open(json_file, 'r') as file:
    api_result = json.load(file)

for key, value in api_result['current_observation']['display_location'].items():
    print(key, value)

print()

for key, value in api_result['current_observation'].items():
    print(key, value)

current = api_result['current_observation']

location = current['display_location']['full'],
weather = current['weather'],
temp = current['temp_c'],
humidity = current['relative_humidity'],
win_deg = current['wind_degrees'],
wind_mph = current['wind_kph'],
dew_pt = current['dewpoint_c'],
precipm = current['precip_today_metric']

print(records)
