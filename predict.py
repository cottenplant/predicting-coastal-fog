import request_conditions
from sklearn.externals import joblib


def predict():
    results = {}
    models = ["dtree_model.pkl", "gbc_model.pkl"]
    input_features = request_conditions.main()
    for model in models:
        clf = joblib.load(model)
        pred = clf.predict(input_features)
        results[model] = pred

    return results


if __name__ == "__main__":
    prediction = predict()
    for model, results in prediction.items():
        if results == [1]:
            print("{} model says.. the fog should roll in tonight!".format(model))
        elif results == [0]:
            print("{} model says... clear skies this evening!".format(model))
