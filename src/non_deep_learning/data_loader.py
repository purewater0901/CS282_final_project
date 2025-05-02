import os
import cv2
import numpy as np
from skimage.feature import hog


def load_images(data_dir, label, img_size=(128 ,128), feature="flatten"):
    images = []
    labels = []
    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if feature=="flatten":
            img = img.flatten() / 255.0 # flatten images for SVM
        elif feature=="hog":
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

def get_images_and_labels(dataset_paths, cfg):
    all_images = []
    all_labels = []
    for dataset_path in dataset_paths:
        if 'Real' in dataset_path:
            print(f'Load Real dataset from {dataset_path}')
            label = 0
        else:
            print(f'Load Fake dataset from {dataset_path}')
            label = 1

        images, labels = load_images(dataset_path, label, feature=cfg.feature)
        all_images.extend(images)
        all_labels.extend(labels)

    return np.array(all_images), np.array(all_labels)

def data_loader(current_dir, cfg):
    root_data_dir = 'data'
    root_data_dir_path = os.path.join(current_dir, root_data_dir)

    subdirs = []
    for name in os.listdir(root_data_dir_path):
        path = os.path.join(root_data_dir_path, name)
        if os.path.isdir(path): 
            subdirs.append(path)

    train_dataset_paths = [os.path.join(subdir_path, 'Train') for subdir_path in subdirs]
    test_dataset_paths = [os.path.join(subdir_path, 'Test') for subdir_path in subdirs]
    X_train, y_train = get_images_and_labels(train_dataset_paths, cfg)
    X_test, y_test = get_images_and_labels(test_dataset_paths, cfg)
   
    return X_train, y_train, X_test, y_test
