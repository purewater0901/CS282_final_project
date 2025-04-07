import numpy as np
import cv2
import os
import argparse
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data_loader import data_loader

SEED = 0

if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default="flatten")
    cfg = parser.parse_args()

    current_dir = os.getcwd()
    X_train, y_train, X_test, y_test = data_loader(current_dir, cfg)

    #model = KNeighborsClassifier(n_neighbors=10)
    model = SVC(kernel='rbf', C=100.0, gamma=0.001)

    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("best parameter: ", grid_search.best_params_)
    print("best score: ", grid_search.best_score_)

    y_pred = grid_search.predict(X_test)
    print("test accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['True', 'Fake']))