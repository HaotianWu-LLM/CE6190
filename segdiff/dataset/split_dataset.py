import numpy as np
import random
import torch
import os
from pathlib import Path

# patient_dir = os.path.join('/home/autodl-tmp/data/brats', 'BraTS2021_MiddleSlice_Train')
patient_dir = './autodl-tmp/data/BraTS2021/All_Data'
case_names = sorted(list(Path(patient_dir).iterdir())) # 1251例

random_seed = 42
np.random.seed(random_seed)
np.random.shuffle(case_names)

num_datapoints = len(case_names)
num_train = int(num_datapoints * 0.8)
num_val = int(num_datapoints * 0.1)
num_test = num_datapoints - num_train - num_val

train_datapoints = case_names[:num_train]
val_datapoints = case_names[num_train:num_train + num_val]
test_datapoints = case_names[num_train + num_val:]


with open("./autodl-tmp/data/BraTS2021/brats_split_training.txt", 'w') as f:
    for s in train_datapoints:
        f.write(str(str(s).split('/')[-1]) + '\n')
with open("./autodl-tmp/data/BraTS2021/brats_split_validing.txt", 'w') as f:
    for s in val_datapoints:
        f.write(str(str(s).split('/')[-1]) + '\n')
with open("./autodl-tmp/data/BraTS2021/brats_split_testing.txt", 'w') as f:
    for s in test_datapoints:
        f.write(str(str(s).split('/')[-1]) + '\n')
