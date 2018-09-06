import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def check_classifier(X, y, test_size=0.3, random_state=101):
    # Standardize scale of features for model comparison
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Try decision tree classifier
    dtree_score = decision_tree(X_train, X_test, y_train, y_test)
    print(dtree_score)

    # Run gamut of classifiers available through scikit-learn and assess accuracy
    acc_frame = run_models(X_train, X_test, y_train, y_test)

    return acc_frame


def decision_tree(X_train, y_train, X_test, y_test):
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    dtree_score = dtree.score(X_test, y_test)

    return dtree_score


def run_models(X_train, y_train, X_test, y_test):
    acc = []
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

    for model in range(len(models)):
        clf = models[model]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc.append(accuracy_score(pred, y_test))

    m = {'Algorithm': model_names, 'Accuracy': acc}

    acc_frame = pd.DataFrame(m)
    acc_frame = acc_frame.set_index('Accuracy').sort_index(ascending=False)

    return acc_frame
