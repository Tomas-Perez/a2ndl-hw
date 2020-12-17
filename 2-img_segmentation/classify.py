import os
import numpy as np
from vgg_base import create_model, MODEL_NAME
from dataset_types import Subdataset, Species


def get_files_in_directory(path, include_folders=False):
    """Get all filenames in a given directory, optionally include folders as well"""
    return [f for f in os.listdir(path) if include_folders or os.path.isfile(os.path.join(path, f))]


def rle_encode(img):
    '''
    img: numpy array, 1 - foreground, 0 - background
    Returns run length as string formatted
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# results dict
submission_dict = {}

# loop through subdatasets and species
for subdataset in Subdataset:
    SUBDATASET = subdataset.value
    for species in Species:
        SPECIES = species.value
        print(f"Classifying {SUBDATASET/VALUE}")


        # get weights
        base_model_exp_dir = f"experiments/{MODEL_NAME}/{SUBDATASET}/{SPECIES}"
        saved_weights = [os.path.join(base_model_exp_dir, f) for f in get_files_in_directory(base_model_exp_dir, include_folders=True)]
        latest_saved_weights_path = max(saved_weights, key=os.path.getctime)
        weights = os.path.join(latest_saved_weights_path, 'best/model')

        # load weights in model
        print(f"\tLoading weights from model: {weights}...")
        model = create_model(256, 256, len(classes))
        model.load_weights(weights)


        # calculate prediction for each image

        # dataset dir
        dataset_dir = f"Development_Dataset/Test_Dev/{SUBDATASET}/{SPECIES}/Images"

        image_filenames = get_files_in_directory(dataset_dir)

        for img_name in image_filenames:
            img = Image.open(f"{dataset_dir}/{img_name}").convert('RGB')
            img = img.resize((256, 256))
            img_array = np.expand_dims(np.array(img), 0) 

            # predict -> (256, 256) with class value
            prediction = model.predict(x=img_array)
            prediction = tf.argmax(prediction, -1) 
            ## Get tensor's value
            ## prediction = tf.keras.backend.get_value(prediction)[0]

            submission_dict[img_name] = {}
            submission_dict[img_name]['shape'] = mask_arr.shape
            submission_dict[img_name]['team'] = SUBDATASET
            submission_dict[img_name]['crop'] = SPECIES
            submission_dict[img_name]['segmentation'] = {}

            # RLE encoding
            # crop
            rle_encoded_crop = rle_encode(prediction == 1)
            # weed
            rle_encoded_weed = rle_encode(prediction == 2)

            submission_dict[img_name]['segmentation']['crop'] = rle_encoded_crop
            submission_dict[img_name]['segmentation']['weed'] = rle_encoded_weed


# Save results into the submission.json file
with open('./predictions/submission.json', 'w') as f:
    json.dump(submission_dict, f)

