import os
from dataset_types import Subdataset, Species
import random

def get_files_in_directory(path, include_folders=False):
    """Get all filenames in a given directory, optionally include folders as well"""
    return [f for f in os.listdir(path) if include_folders or os.path.isfile(os.path.join(path, f))]

if __name__ == "__main__":
    SEED = 9999
    random.seed(SEED)

    DATASET_DIR = "Development_Dataset/Training"
    VALIDATION_SPLIT = 0.2
    for sd in Subdataset:
        for species in Species:
            this_dataset_dir = os.path.join(DATASET_DIR, sd.value, species.value)
            image_names = get_files_in_directory(os.path.join(this_dataset_dir, "Images"))
            val_list = []
            train_list = []
            for i in image_names:
                if random.uniform(0, 1) < VALIDATION_SPLIT:
                    val_list.append(i)
                else:
                    train_list.append(i)

            this_split_dir = os.path.join(this_dataset_dir, "Splits")
            
            if not os.path.exists(this_split_dir):
                os.makedirs(this_split_dir)

            with open(os.path.join(this_split_dir, "train.txt"), "w") as f:
                for i in train_list:
                    f.write(os.path.splitext(i)[0] + "\n")

            with open(os.path.join(this_split_dir, "val.txt"), "w") as f:
                for i in val_list:
                    f.write(os.path.splitext(i)[0] + "\n")
