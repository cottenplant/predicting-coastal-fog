# Coastal Fog Prediction
## Santa Monica, CA

### predict-fog.py
Output:
```
Current Conditions:

loc Santa Monica, CA
dewpoint_f 56
pressure_mb 1007
temp_f 58.1
precip_today_in 
wind_degrees 205
wind_mph 0

dtree_model.pkl model says... clear skies this evening!
gbc_model.pkl model says... clear skies this evening!
```
## It's easy
- First, current weather conditions are pulled via Wunderground.com API
```
def get_from_api(city):
    api_key = '<API_key>'
    url = 'http://api.wunderground.com/api/{}/conditions/q/{}.json'
    response = urllib.request.urlopen(url.format(api_key, city))
    data = json.loads(response.read())

    return data
```
- JSON response decoded into Python dict
```
def get_from_json(result, meas):
    obs = {}
    current = result['current_observation']
    obs['loc'] = current['display_location']['full']
    for key in meas:
        obs[key] = current[key]

    return obs
```
- Feature array prepared for use in model. Windspeed, temperature, dewpoint, pressure, and daily precipation totals are funneled in
```
def make_feature_array(obs):
    feat_arr = []
    features = ['wind_degrees', 'wind_mph', 'temp_f', 'dewpoint_f', 'pressure_mb', 'precip_today_in']
    for feat in features:
        if obs[feat] == '':
            feat_arr.append(0)
        else:
            feat_arr.append(float(obs[feat]))

    return feat_arr
```
- Currently, I am comparing Decision Tree and Gradient Boosting Classifiers, hence the two outputs. Both had different advantages in terms of precision / recall
```
def predict(models, input_features):
    predictions = {}
    for classifier in models:
        clf = joblib.load(classifier)
        pred = clf.predict(input_features)
        predictions[classifier] = pred

    return predictions


def main():
    input_array = get_input_array()
    models_list = ["dtree_model.pkl", "gbc_model.pkl"]
    prediction_results = predict(models_list, input_array)

    return prediction_results
```
For more background info, check out [the documentation](https://samcotten.github.io/coastal-climate-analysis).
