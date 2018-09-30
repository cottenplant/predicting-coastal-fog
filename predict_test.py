import request_conditions
from sklearn.externals import joblib


def predict():
    model = "dtree_model.pkl"
    input_features = request_conditions.main()
    clf = joblib.load(model)
    pred = clf.predict(input_features)

    return pred


if __name__ == "__main__":
    prediction = predict()
    print(prediction)
