import numpy as np
import cv2
import os
import argparse
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

SEED = 0

def load_images(data_dir, label, img_size=(128 ,128), features="flatten"):
    images = []
    labels = []
    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if features=="flatten":
            img = img.flatten() / 255.0 # flatten images for SVM
        elif features=="hog":
            img = hog(
                img,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(2,2),
                block_norm='L2-Hys',
            )
        elif features=="lbp":
            P = 8.0
            R = 1.0
            lbp = local_binary_pattern(img, P, R, method="uniform")
            (img, _) = np.histogram(lbp.ravel(),bins=np.arange(0, P + 3), range=(0, P + 2))
            img = img.astype("float")
            img /= (img.sum() + 1e-6)
        else:
            raise NotImplementedError("Feature extraction method not implemented")

        images.append(img)
        labels.append(label)
    return images, labels

if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default="flatten")
    cfg = parser.parse_args()

    current_dir = os.getcwd()
    data_dir = 'data'
    data_dir_path = os.path.join(current_dir, data_dir)

    real_face_data_path = os.path.join(data_dir_path, 'Real_split')
    style_gan_data_path = os.path.join(data_dir_path, 'StyleGAN_split')
    firefly_data_path = os.path.join(data_dir_path, 'firefly_split')

    real_face_train_data_path = os.path.join(real_face_data_path, 'Train')
    real_face_test_data_path = os.path.join(real_face_data_path, 'Test')
    style_gan_train_data_path = os.path.join(style_gan_data_path, 'Train')
    style_gan_test_data_path = os.path.join(style_gan_data_path, 'Test')
    firefly_train_data_path = os.path.join(firefly_data_path, 'Train')
    firefly_test_data_path = os.path.join(firefly_data_path, 'Test')

    real_face_train_images, real_face_train_labels = load_images(real_face_train_data_path, 0, features=cfg.feature)
    real_face_test_images, real_face_test_labels = load_images(real_face_test_data_path, 0, features=cfg.feature)
    style_gan_train_images, style_gan_train_labels = load_images(style_gan_train_data_path, 1, features=cfg.feature)
    style_gan_test_images, style_gan_test_labels = load_images(style_gan_test_data_path, 1, features=cfg.feature)
    firefly_train_images, firefly_train_labels = load_images(firefly_train_data_path, 1, features=cfg.feature)
    firefly_test_images, firefly_test_labels = load_images(firefly_test_data_path, 1, features=cfg.feature)

    X_train = np.array(real_face_train_images + style_gan_train_images + firefly_train_images)
    y_train = np.array(real_face_train_labels + style_gan_train_labels + firefly_train_labels)

    X_test = np.array(real_face_test_images + style_gan_test_images + firefly_test_images)
    y_test = np.array(real_face_test_labels + style_gan_test_labels + firefly_test_labels)

    k_neighbors = KNeighborsClassifier(n_neighbors=10)

    param_grid = {
        'n_neighbors': [10, 50, 100, 200, 300, 400, 500, 1000],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    grid_search = GridSearchCV(
        k_neighbors, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("best parameter: ", grid_search.best_params_)
    print("best score: ", grid_search.best_score_)

    y_pred = grid_search.predict(X_test)
    print("test accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['True', 'Fake']))