import clean_data
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


def main(rs):
    # Import cleaned data set
    df_smo = clean_data.init()
    X = df_smo[['mdir', 'mspd', 'mtmp', 'mdew', 'mpressure', 'precipm']]
    y = df_smo['fog']

    # Data Preprocessing
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)

    # Try all Scikit-Learn Classifiers
    results = get_model(X_train, X_test, y_train, y_test)
    results = results.set_index('Algorithm')
    print(results)

    return results


def decision_tree_result(X_train, X_test, y_train, y_test):
    algo = 'Decision Tree Classifier'
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    pred = dtree.predict(X_test)
    score = dtree.score(X_test, y_test)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    return algo, score, precision, recall


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

    for model in range(len(models)):
        clf = models[model]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc.append(accuracy_score(y_test, pred))
        prec.append(precision_score(y_test, pred))
        rec.append(recall_score(y_test, pred))

    m = {'Algorithm': model_names, 'Accuracy': acc, 'Precision': prec, 'Recall': rec}
    report = pd.DataFrame(m)

    return report


if __name__ == "__main__":
    main(rs=42)
