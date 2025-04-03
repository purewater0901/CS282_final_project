# CS282 Final Project (Fake image detection)

This repository contains the code for the CS282 Final project.

## 1. Dataset Setup

### Step 1: Download the dataset

Download the dataset (ZIP file) from [this Kaggle link](https://www.kaggle.com/datasets/katsuyamucb/madde-dataset).

### Step 2: Create the `data/` directory and extract the dataset

```bash
git clone git@github.com:purewater0901/CS282_final_project.git
cd CS282_final_project
mkdir data && cd data
unzip <path_to_downloaded_zip>
```

## 2. Running the Code

> ⚠️ Make sure you have downloaded the dataset as described above before running the code.


### 1. Environment Setup
```bash
conda create -n cs282 python=3.10 pip
conda activate cs282
pip install scikit-learn numpy opencv-python pillow matplotlib scikit-image
```

### 2. Running code

- SVM

```bash
cd CS282_final_project
python src/svm.py
```