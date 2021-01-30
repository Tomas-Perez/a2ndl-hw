import os
import tensorflow as tf
import numpy as np
import json
import math
from labels_dict import labels_dict
from PIL import Image

def k_split(idx, data, val_split, which_subset):
    len_val = math.ceil(len(data) * val_split)
    if which_subset == 'training':
        left_side = data[:idx * len_val]
        right_side = data[idx * len_val + len_val:]
        return left_side + right_side
    elif which_subset == 'validation':
        return data[idx * len_val : idx * len_val + len_val]
    else:
        raise RuntimeError(f'Unknown subset {which_subset}')

# Dataset of image and questions which allows for division in k-folds
class VQADataset(tf.keras.utils.Sequence):

    def __init__(self, dataset_dir, which_subset, tokenized_texts, num_classes, img_generator=None, 
        img_preprocessing_function=None, img_out_shape=[256, 256], validation_split=0.2, k_idx=0):
        
        with open(os.path.join(dataset_dir, 'train_questions_annotations.json')) as f:
            self.annotations = k_split(k_idx, list(json.load(f).items()), validation_split, which_subset)

        self.which_subset = which_subset
        self.dataset_dir = dataset_dir
        self.img_generator = img_generator
        self.preprocessing_function = img_preprocessing_function
        self.out_shape = img_out_shape
        self.tokenized_texts = tokenized_texts
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        curr_annotation = self.annotations[index][1]
        curr_filename = curr_annotation["image_id"] + ".png"
        curr_question = self.tokenized_texts[self.annotations[index][0]]
        correct_idx = labels_dict[curr_annotation["answer"]]
        
        # Read Image and perform augmentation if necessary
        img = Image.open(os.path.join(self.dataset_dir, 'Images', curr_filename)).convert('RGB')
        
        img = img.resize(self.out_shape)
        img_arr = np.array(img)

        if self.which_subset == 'training' and self.img_generator is not None:
            img_t = self.img_generator.get_random_transform(img_arr.shape)
            img_arr = self.img_generator.apply_transform(img_arr, img_t)
        
        if self.preprocessing_function is not None:
            img_arr = self.preprocessing_function(img_arr)

        return (img_arr, curr_question), correct_idx