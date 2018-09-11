import request_conditions
from sklearn.externals import joblib


def predict():
    input_features = request_conditions.main()
    clf = joblib.load("gbc_model.pkl")
    pred = clf.predict(input_features)

    return pred


if __name__ == "__main__":
    prediction = predict()
    if prediction == [1]:
        print("The fog should roll in today!")
    elif prediction == [0]:
        print("Clear skies this evening!")
