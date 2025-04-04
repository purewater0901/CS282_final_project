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
pip install scikit-learn numpy pandas opencv-python pillow matplotlib scikit-image wandb IPython kaggle
pip install torch torchvision torchaudio
pip install timm
```

### 2. Running code

- Classical machine learning approach (K-Nearest, SVM, Random Forest)

```bash
python src/non_deep_learning/ml_baselines.py --feature=<feature>
```

Here `<feature>` can be `flatten` or `hog`. We set `flatten` as a default value in this problem.
