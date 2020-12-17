import tensorflow as tf
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(model, valid_dataset, num_classes):
    # Assign a color to each class
    evenly_spaced_interval = np.linspace(0, 1, 2)
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    iterator = iter(valid_dataset)

    fig, ax = plt.subplots(1, 3, figsize=(8, 8))
    image, target = next(iterator)

    image = image[0]
    target = target[0, ..., 0]

    out_sigmoid = model.predict(x=tf.expand_dims(image, 0))

    # Get predicted class as the index corresponding to the maximum value in the vector probability
    # predicted_class = tf.cast(out_sigmoid > score_th, tf.int32)
    # predicted_class = predicted_class[0, ..., 0]
    predicted_class = tf.argmax(out_sigmoid, -1)

    predicted_class = predicted_class[0, ...]

    # Assign colors (just for visualization)
    target_img = np.zeros([target.shape[0], target.shape[1], 3])
    prediction_img = np.zeros([target.shape[0], target.shape[1], 3])

    target_img[np.where(target == 0)] = [0, 0, 0]
    for i in range(1, num_classes):
        target_img[np.where(target == i)] = np.array(colors[i-1])[:3] * 255

    prediction_img[np.where(predicted_class == 0)] = [0, 0, 0]
    for i in range(1, num_classes):
        prediction_img[np.where(predicted_class == i)] = np.array(colors[i-1])[:3] * 255

    ax[0].imshow(np.uint8(image))
    ax[1].imshow(np.uint8(target_img))
    ax[2].imshow(np.uint8(prediction_img))

    plt.show()