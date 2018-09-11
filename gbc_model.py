import clean_data

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


def gbc():
    df_smo = clean_data.init()
    X = df_smo[['mdir', 'mspd', 'mtmp', 'mdew', 'mpressure', 'precipm']]
    y = df_smo['fog']
    sc = StandardScaler()
    X = sc.fit_transform(X)
    clf = GradientBoostingClassifier()
    clf.fit(X, y)
    joblib.dump(clf, "gbc_model.pkl")


if __name__ == "__main__":
    gbc()
