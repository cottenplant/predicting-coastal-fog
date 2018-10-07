import request_conditions
from sklearn.externals import joblib


def predict(models):
    predictions = {}
    input_features = request_conditions.main()
    for classifier in models:
        clf = joblib.load(classifier)
        pred = clf.predict(input_features)
        predictions[classifier] = pred

    return predictions


def main():
    models_list = ["dtree_model.pkl", "gbc_model.pkl"]
    prediction_results = predict(models_list)

    return prediction_results


if __name__ == "__main__":
    results = main()
    for model, results in results.items():
        if results == [1]:
            print("{} model says.. the fog should roll in tonight!".format(model))
        elif results == [0]:
            print("{} model says... clear skies this evening!".format(model))
