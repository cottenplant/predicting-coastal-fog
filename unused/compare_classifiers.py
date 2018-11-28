import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def prepare_data(df):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
    random_state=42)

    return X_train, X_test, y_train, y_test


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
    prec = []
    rec = []
    conf_mat = []
    class_rep = []

    for model in range(len(models)):
        clf = models[model]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc.append(accuracy_score(y_test, pred))
        prec.append(precision_score(y_test, pred))
        rec.append(recall_score(y_test, pred))
        conf_mat.append(confusion_matrix(y_test, pred))

    m = {'Algorithm': model_names, 'Accuracy': acc, 'Precision': prec, 'Recall': rec,
         'Confusion Matrix': conf_mat}
    report = pd.DataFrame(m)

    return report
