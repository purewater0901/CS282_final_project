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
pip install -r requirements.txt
```

### Step 3: Download the dataset
> [!Warning]
> Since a small portion of our dataset is owned by the organization the authors belong to and is not allowed to publically share, the following dataset is set to be private for now.
> For reproduction, please reach out to `k.masaki@berkeley.edu` to get access the dataset. Sorry for the inconvenience.

First you need to create/login your Kaggle dataset, and once you got the access to our dataset, please run
```bash
cd CS282_final_project
mkdir data
echo 'export PATH=/opt/pytorch/bin:$PATH' >> ~/.bashrc # To run pytorch on the dataset
```

Run a downloading script;
```python
import kagglehub
import os

# Set target download directory
os.environ["KAGGLEHUB_CACHE_DIR"] = os.path.expanduser("~/CS282_final_project/data")
# Download latest version
path = kagglehub.dataset_download("katsuyamucb/madde-dataset")

print("Path to dataset files:", path)
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
