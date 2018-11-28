import clean_data

from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib


def decision_tree():
    df_smo = clean_data.init()
    X = df_smo[['mdir', 'mspd', 'mtmp', 'mdew', 'mpressure', 'precipm']]
    y = df_smo['fog']
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    joblib.dump(clf, "dtree_model.pkl")


if __name__ == "__main__":
    decision_tree()
