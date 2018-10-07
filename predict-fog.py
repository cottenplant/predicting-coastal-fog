import json
import urllib.request
from sklearn.externals import joblib


def get_city_name(query):
    base_url = 'http://autocomplete.wunderground.com/aq?query={}'
    response = urllib.request.urlopen(base_url.format(query))
    data = json.loads(response.read())

    return data


def get_from_api(city):
    api_key = '8940b5f3356ab273'
    url = 'http://api.wunderground.com/api/{}/conditions/q/{}.json'
    response = urllib.request.urlopen(url.format(api_key, city))
    data = json.loads(response.read())

    return data


def get_from_json(result, meas):
    obs = {}
    current = result['current_observation']
    obs['loc'] = current['display_location']['full']
    for key in meas:
        obs[key] = current[key]

    return obs


def make_feature_array(obs):
    feat_arr = []
    features = ['wind_degrees', 'wind_mph', 'temp_f', 'dewpoint_f', 'pressure_mb', 'precip_today_in']
    for feat in features:
        if obs[feat] == '':
            feat_arr.append(0)
        else:
            feat_arr.append(float(obs[feat]))

    return feat_arr


def get_input_array():
    input_city = "CA/Santa_monica"
    measurements = ['dewpoint_f', 'pressure_mb', 'temp_f', 'precip_today_in', 'wind_degrees', 'wind_mph']
    city_data = get_from_api(input_city)
    observations = get_from_json(city_data, measurements)
    print('Current Conditions:\n')
    for feat, label in observations.items():
        print(feat, label)
    print()
    input_array = make_feature_array(observations)

    return [input_array]


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


if __name__ == "__main__":
    results = main()
    for model, results in results.items():
        if results == [1]:
            print("{} model says.. the fog should roll in tonight!".format(model))
        elif results == [0]:
            print("{} model says... clear skies this evening!".format(model))
