# CS282 Final Project (Fake image detection)

This repository contains the code for the CS282 Final project.

## 1. Setup

### Step 1: Clone the repository

```bash
git clone git@github.com:purewater0901/CS282_final_project.git
cd CS282_final_project
```

### Step 2: Set up the environment

```bash
conda create -n cs282 python=3.10 pip
conda activate cs282
pip install scikit-learn numpy pandas opencv-python pillow matplotlib scikit-image wandb IPython kaggle
pip install torch torchvision torchaudio transformers
pip install timm peft
```

### Step 3: Create the `data/` folder and download the data

```bash
cd CS282_final_project
mkdir data && cd data
kaggle datasets download katsuyamucb/madde-dataset
unzip madde-dataset.zip
```

## 2. Running the Code

> ⚠️ Make sure you have downloaded the dataset as described above before running the code.

### 1. Classical machine learning approach (K-Nearest, SVM, Random Forest)

```bash
python src/non_deep_learning/ml_baselines.py --feature=<feature>
```

Here `<feature>` can be `flatten` or `hog`. We set `flatten` as a default value in this problem.

### 2.Deep Learning based approach

```bash
python src/main.py --config=config/<filename>
```

See `config` folder for the model that you want to use.

Example:

```bash
python src/main.py --config config/efficient_net_full_tune.yaml
```
