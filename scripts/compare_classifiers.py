import pandas as pd
import preprocess_data

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import recall_score, confusion_matrix, roc_curve, auc

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings(action='ignore')


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
    conf_mat = []
    roc_auc = []

    for model in range(len(models)):
        clf = models[model]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc.append(accuracy_score(y_test, pred))
        prec.append(precision_score(y_test, pred))
        rec.append(recall_score(y_test, pred))
        conf_mat.append(confusion_matrix(y_test, pred)[1][1])
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred)
        roc_auc.append(auc(false_positive_rate, true_positive_rate))

    m = {'Algorithm': model_names, 'Accuracy': acc, 'Precision': prec,
         'Recall': rec, 'Confusion_tp': conf_mat, 'ROC AUC': roc_auc}
    report = pd.DataFrame(m)

    return report


def run_search(dataset, n):
    df = pd.DataFrame()
    for i in range(0, n + 1):
        print("\nTEST # {}".format(i))
        results = main(dataset, i)
        df = pd.concat([df, results])
    summary = df.groupby('Algorithm').mean()
    print("\n\n===SUMMARY===")
    print(summary)
    summary.to_csv(dataset + '_compare_classifiers_summary.csv')


def main(dataset, rs=None):
    if dataset == 'ksmo':
        df = preprocess_data.ksmo()
        features = ['meanwdird', 'meanwindspdm', 'meantempm', 'meandewptm',
                    'meanpressurem', 'mhumidity', 'maxtempm', 'precipm']
        target = ['fog']
    elif dataset == 'klax':
        df = preprocess_data.klax()
        features = ['TEMP', 'DEWP', 'VISIB', 'WDSP', 'PRCP']
        target = ['FOG']
    else:
        print("Error! Pass in dataset as kwarg [ksmo, klax].")
    X = df[features]
    y = df[target]

    # Data Preprocessing
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=rs)
    # Try all Scikit-Learn Classifiers
    results = get_model(X_train, X_test, y_train, y_test)
    results = results.set_index('Algorithm')
    print(results)

    return results


if __name__ == "__main__":
    # main('klax', None)
    run_search('klax', 10)
