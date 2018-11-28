import json
import urllib.request
from sklearn.externals import joblib


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

def make_feature_array(obs, features, dataset):
    feat_arr = []
    if dataset == 'ksmo':
        obs['relative_humidity'] = obs['relative_humidity'].strip('%')
    for feat in features:
        if obs[feat] == '':
            feat_arr.append(0)
        else:
            feat_arr.append(float(obs[feat]))
    print('Current Conditions:\n')
    for feat, label in obs.items():
        print(feat, label)
    print()

    return [feat_arr]


def predict(classifier, input_features):
    clf = joblib.load(classifier)
    pred = clf.predict(input_features)

    return pred


def main(dataset):
    # Static Vars
    input_city = "CA/Santa_monica"
    if dataset == 'klax':
        parameters = ['temp_f', 'dewpoint_f', 'visibility_mi', 'wind_mph',
                      'precip_today_in']
        model = "klax_GaussianNB_model.pkl"
    elif dataset == 'ksmo':
        parameters = ['wind_degrees', 'wind_mph', 'temp_f', 'dewpoint_f',
                'pressure_mb', 'relative_humidity', 'precip_today_in']
        model = "ksmo_DTree_model.pkl"
    city_data = get_from_api(input_city)
    observations = get_from_json(city_data, parameters)
    input_array = make_feature_array(observations, parameters, dataset)
    prediction_results = predict(model, input_array)
    if prediction_results == [1]:
        print("Decision Tree model says... the fog should roll in tonight!")
    elif prediction_results == [0]:
        print("Decision Tree model says... clear skies this evening!")

    return prediction_results


if __name__ == "__main__":
    stations = ['ksmo', 'klax']
    for station in stations:
        result = main(station)
        print("(measuring from {} station)\n".format(station))
