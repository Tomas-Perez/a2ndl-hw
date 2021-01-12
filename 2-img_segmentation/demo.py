import os
import tensorflow as tf
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from vgg_base import create_model, CustomDataset, MODEL_NAME
from dataset_types import Subdataset, Species
from tensorflow.keras.applications.vgg16 import preprocess_input 
from PIL import Image
from mask_colors import BACKGROUND_1, BACKGROUND_0, WEED, CROP
from files_in_dir import get_files_in_directory

dataset_base = "Development_Dataset/Training"
SUBDATASET = Subdataset.BIPBIP.value
SPECIES = Species.HARICOT.value

img_h, img_w = 256, 256

dataset_dir = os.path.join(dataset_base, SUBDATASET, SPECIES)

image_name = "Bipbip_haricot_im_00391"

img_path = os.path.join(dataset_dir, "Images", f"{image_name}.jpg")
mask_path = os.path.join(dataset_dir, "Masks", f"{image_name}.png")

img = Image.open(img_path).resize([img_h, img_w])
mask = Image.open(mask_path).resize([img_h, img_w], resample=Image.NEAREST)
mask_arr = np.array(mask)

# Convert RGB mask for each class to numbers from 0 to 2
new_mask_arr = np.zeros(mask_arr.shape[:2], dtype=mask_arr.dtype)
new_mask_arr = np.expand_dims(new_mask_arr, -1)

new_mask_arr[np.where(np.all(mask_arr == BACKGROUND_0, axis=-1))] = 0
new_mask_arr[np.where(np.all(mask_arr == BACKGROUND_1, axis=-1))] = 0
new_mask_arr[np.where(np.all(mask_arr == CROP, axis=-1))] = 1
new_mask_arr[np.where(np.all(mask_arr == WEED, axis=-1))] = 2

new_mask_arr = np.float32(new_mask_arr)

num_classes = 3

# get weights
base_model_exp_dir = f"experiments/{MODEL_NAME}/{SUBDATASET}/{SPECIES}"
saved_weights = [os.path.join(base_model_exp_dir, f) for f in get_files_in_directory(base_model_exp_dir, include_folders=True)]
latest_saved_weights_path = max(saved_weights, key=os.path.getctime)
weights = os.path.join(latest_saved_weights_path, 'best/model')

print(weights)

model = create_model(img_h, img_w, num_classes)
model.load_weights(weights)

img_array = np.array(img)
img_array = preprocess_input(img_array)

# predict -> (256, 256) with class value
prediction = model.predict(x=np.expand_dims(img_array, 0))
prediction = tf.argmax(prediction, -1) 
## Get tensor's value
prediction = np.matrix(tf.keras.backend.get_value(prediction))

# Assign a color to each class
evenly_spaced_interval = np.linspace(0, 1, 2)
colors = [cm.rainbow(x) for x in evenly_spaced_interval]

fig, ax = plt.subplots(1, 3, figsize=(8, 8))

target = new_mask_arr[..., 0]

# Assign colors (just for visualization)
target_img = np.zeros([target.shape[0], target.shape[1], 3])
prediction_img = np.zeros([target.shape[0], target.shape[1], 3])

target_img[np.where(target == 0)] = [0, 0, 0]
for i in range(1, num_classes):
    target_img[np.where(target == i)] = np.array(colors[i-1])[:3] * 255

prediction_img[np.where(prediction == 0)] = [0, 0, 0]
for i in range(1, num_classes):
    prediction_img[np.where(prediction == i)] = np.array(colors[i-1])[:3] * 255

ax[0].imshow(img)
ax[1].imshow(np.uint8(target_img))
ax[2].imshow(np.uint8(prediction_img))

plt.show()
