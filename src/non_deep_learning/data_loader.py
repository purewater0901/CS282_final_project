import os
import cv2
import numpy as np
from skimage.feature import hog


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
        else:
            raise NotImplementedError("Feature extraction method not implemented")

        images.append(img)
        labels.append(label)
    return images, labels

def data_loader(current_dir, cfg):
    data_dir = 'data'
    data_dir_path = os.path.join(current_dir, data_dir)

    real_face_data_path = os.path.join(data_dir_path, 'Real_split')
    fake_face_data_path = os.path.join(data_dir_path, 'All_fakes_split')

    real_face_train_data_path = os.path.join(real_face_data_path, 'Train')
    real_face_test_data_path = os.path.join(real_face_data_path, 'Test')
    fake_face_train_data_path = os.path.join(fake_face_data_path, 'Train')
    fake_face_test_data_path = os.path.join(fake_face_data_path, 'Test')

    real_face_train_images, real_face_train_labels = load_images(real_face_train_data_path, 0, features=cfg.feature)
    real_face_test_images, real_face_test_labels = load_images(real_face_test_data_path, 0, features=cfg.feature)
    fake_face_train_images, fake_face_train_labels = load_images(fake_face_train_data_path, 1, features=cfg.feature)
    fake_face_test_images, fake_face_test_labels = load_images(fake_face_test_data_path, 1, features=cfg.feature)

    X_train = np.array(real_face_train_images + fake_face_train_images)
    y_train = np.array(real_face_train_labels + fake_face_train_labels)

    X_test = np.array(real_face_test_images + fake_face_test_images)
    y_test = np.array(real_face_test_labels + fake_face_test_labels)

    return X_train, y_train, X_test, y_test