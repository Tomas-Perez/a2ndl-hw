import os
import random
import numpy as np
from datetime import datetime
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set TF logging to ERROR, needs to be done before importing TF
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from classes import classes
import callbacks

"""
Model based on VGG with imagenet weights, used for transfer learning and finetuning.
"""

MODEL_NAME = 'MaskDetection-Transfer'

def VGG_transfer_model(img_h, img_w, num_classes, train_mode=False):
    vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))

    if train_mode:
        if FINETUNING:
            freeze_until = 15

            for layer in vgg.layers[:freeze_until]:
                layer.trainable = False
        else:
            vgg.trainable = False

    model = tf.keras.Sequential()
    model.add(vgg)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    return model

def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    AUGMENT_DATA = True
    CHECKPOINTS = False
    EARLY_STOP = True
    FINETUNING = True
    SAVE_BEST = True
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 1e-4

    img_h = 256
    img_w = 256
        
    # Set global seed for all internal generators, this should make all randomization reproducible
    import signal
    SEED = signal.SIGSEGV.value # Set SEED to SEG_FAULT code (11)
    set_seeds(SEED)

    preprocess_input = tf.keras.applications.vgg16.preprocess_input

    # Data generator
    # Split training and validation data automatically
    if AUGMENT_DATA:
        train_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=10,
            height_shift_range=10,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant',
            cval=0,
            rescale=1./255,
            validation_split=VALIDATION_SPLIT,
            preprocessing_function=preprocess_input,
        )
    else:
        train_gen = ImageDataGenerator(
            rescale=1./255,
            validation_split=VALIDATION_SPLIT, 
            preprocessing_function=preprocess_input,
        )

    valid_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT, 
        preprocessing_function=preprocess_input,
    )

    # Training and validation datasets
    dataset_dir = "MaskDataset"

    bs = 8

    num_classes = len(classes)

    training_dir = f"{dataset_dir}/training-structured"

    def flow(generator, subset):
        return generator.flow_from_directory(
            training_dir,
            target_size=(img_h, img_w),
            color_mode='rgb',
            batch_size=bs,
            classes=classes,
            class_mode='categorical',
            shuffle=True,
            subset=subset,
        )

    train_flow = flow(train_gen, 'training')
    validation_flow = flow(valid_gen, 'validation')

    def dataset_from_flow(flow):
        return tf.data.Dataset.from_generator(
            lambda: flow,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, img_h, img_w, 3], [None, num_classes])
        ).repeat()

    train_dataset = dataset_from_flow(train_flow)
    validation_dataset = dataset_from_flow(validation_flow)

    # --------- Training ---------

    model = VGG_transfer_model(img_h, img_w, num_classes, train_mode=True)

    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    metrics = ['accuracy']

    # Compile Model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks_list = []

    exps_dir = "experiments"
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    now = datetime.now().strftime('%b%d_%H-%M-%S')

    model_dir = os.path.join(exps_dir, MODEL_NAME)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    exp_dir = os.path.join(model_dir, str(now))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    callbacks_list = []

    # Model checkpoint
    if CHECKPOINTS:
        callbacks_list.append(callbacks.checkpoints(exp_dir))

    # Early stopping
    if EARLY_STOP:
        callbacks_list.append(callbacks.early_stopping(patience=7))

    # Save best model
    # ----------------
    if SAVE_BEST:
        callbacks_list.append(callbacks.save_best(exp_dir))

    model.fit(x=train_dataset,
        epochs=100,
        steps_per_epoch=len(train_flow),
        validation_data=validation_dataset,
        validation_steps=len(validation_flow),
        callbacks=callbacks_list,
    )