import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from PIL import Image

from csv_generator import create_csv

from transfer_model import VGG_transfer_model as network_model, MODEL_NAME
# from homebrew_model import homebrew_model as network_model, MODEL_NAME
from classes import classes

dataset_dir = "MaskDataset"

checkpoint_timestamp = "Nov11_22-44-20"
model = network_model(256, 256, len(classes))
model.load_weights(f'experiments/{MODEL_NAME}/{checkpoint_timestamp}/best/model')

# for each image in test folder, calculate prediction and add to results

results = {}
image_filenames = next(os.walk(f"{dataset_dir}/test"))[2]

# make a square image while keeping aspect ratio and filling with fill_color
def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

for image_name in image_filenames:
    img = Image.open(f"{dataset_dir}/test/{image_name}").convert('RGB')
    # NN input is 256, 256
    img = img.resize((256, 256))
    img_array = np.expand_dims(np.array(img), 0) 
    # Normalize
    img_array = img_array / 255.

    # Get prediction
    softmax = model.predict(x=img_array)
    # Get predicted class (index with max value)
    prediction = tf.argmax(softmax, 1)
    # Get tensor's value
    prediction = tf.keras.backend.get_value(prediction)[0]

    results[image_name] = prediction

create_csv(results)