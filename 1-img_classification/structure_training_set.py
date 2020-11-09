import random
import json
import os
from classes import classes
from pathlib import Path
import shutil

dataset_dir = "MaskDataset"
source_dir = f"{dataset_dir}/training"
destination_dir = f"{dataset_dir}/training-structured"

with open(f"{dataset_dir}/train_gt.json") as f:
    ground_truth = json.load(f)

def get_files_in_directory(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

training_dirs = []

for c in classes:
    training_dirs.append(f"{destination_dir}/{c}")

for d in training_dirs:
    Path(d).mkdir(parents=True, exist_ok=True)

for f in get_files_in_directory(source_dir):
    class_idx = ground_truth[f]
    shutil.copy2(f"{source_dir}/{f}", training_dirs[class_idx])