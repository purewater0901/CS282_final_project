import numpy as np
import os
import argparse
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from data_loader import data_loader

SEED = 42

if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default="flatten")
    cfg = parser.parse_args()

    current_dir = os.getcwd()
    X_train, y_train, X_test, y_test = data_loader(current_dir, cfg)

    linear_svm = SVC(kernel='linear', C=100.0)
    if cfg.feature == "hog":
        rbf_svm = SVC(kernel='rbf', C=100.0, gamma=0.001)
    else:
        rbf_svm = SVC(kernel='rbf', C=100.0, gamma=0.0001)
    random_forest = RandomForestClassifier(n_estimators=10, random_state=SEED)
    k_neighbors = KNeighborsClassifier(n_neighbors=10)

    # Train and evaluate Linear SVM
    linear_svm.fit(X_train, y_train)
    y_pred = linear_svm.predict(X_test)
    print('Linear SVM Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['True Image', 'Fake Image']))

    # Train and evaluate Non-Linear SVM
    rbf_svm.fit(X_train, y_train)
    y_pred = rbf_svm.predict(X_test)
    print('Nonlinear SVM Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['True Image', 'Fake Image']))

    # Train and evaluate Random Forest
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    print('Random Forest Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['True Image', 'Fake Image']))

    # Train and evaluate Random Forest
    k_neighbors.fit(X_train, y_train)
    y_pred = k_neighbors.predict(X_test)
    print('K Neighbors Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['True Image', 'Fake Image']))
