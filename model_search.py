import clean_data
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def init():
    df_smo = clean_data.init()
    X = df_smo[['mdir', 'mspd', 'mtmp', 'mdew', 'mpressure', 'precipm']]
    y = df_smo['fog']

    sc = StandardScaler()
    X = sc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    dtree_score = dtree(X_train, X_test, y_train, y_test)
    print("Decision tree accuracy:", dtree_score)

    accuracy_df = get_model(X_train, X_test, y_train, y_test)

    print(accuracy_df)


def dtree(X_train, X_test, y_train, y_test):
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)

    return dtree.score(X_test, y_test)


def get_model(X_train, X_test, y_train, y_test):
    models = [LogisticRegression(),
              LinearSVC(),
              SVC(kernel='rbf'),
              KNeighborsClassifier(),
              RandomForestClassifier(),
              DecisionTreeClassifier(),
              GradientBoostingClassifier(),
              GaussianNB()]

    model_names = ['Logistic Regression',
                   'Linear SVM',
                   'rbf SVM',
                   'K-Nearest Neighbors',
                   'Random Forest Classifier',
                   'Decision Tree',
                   'Gradient Boosting Classifier',
                   'Gaussian NB']

    acc = []

    for model in range(len(models)):
        clf = models[model]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc.append(precision_score(pred, y_test))

    m = {'Algorithm': model_names, 'Precision': acc}

    acc_frame = pd.DataFrame(m)
    acc_frame = acc_frame.set_index('Precision').sort_index(ascending=False)

    return acc_frame


if __name__ == "__main__":
    init()
