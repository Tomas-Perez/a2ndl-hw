import os
import random
import numpy as np
from datetime import datetime
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set TF logging to ERROR, needs to be done before importing TF
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# local imports
from classes import classes
import callbacks
from csv_generator import create_csv

MODEL_NAME = 'MaskDetection-Transfer'

def VGG_transfer_model(img_h, img_w, num_classes, finetuning=False):
    vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))

    if finetuning:
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

    loss = tf.keras.losses.CategoricalCrossentropy()
    lr = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    metrics = ['accuracy']

    # Compile Model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
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

    img_h = 256
    img_w = 256
        
    # Set global seed for all internal generators, this should make all randomization reproducible
    import signal
    SEED = signal.SIGSEGV.value
    set_seeds(SEED)

    preprocess_input = tf.keras.applications.vgg16.preprocess_input

    # Data generator
    # Single image generator, splitting the entire dataset in training and validation
    if AUGMENT_DATA:
        data_gen = ImageDataGenerator(
            # rotation_range=10,
            width_shift_range=10,
            height_shift_range=10,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant',
            cval=0,
            rescale=1./255,
            validation_split=0.2,
            preprocessing_function=preprocess_input,
        )
    else:
        data_gen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2, 
            preprocessing_function=preprocess_input,
        )

    # Training and validation datasets
    dataset_dir = "MaskDataset"

    bs = 8
    img_h = 256
    img_w = 256

    num_classes = len(classes)

    training_dir = f"{dataset_dir}/training-structured"

    def flows(generator):
        def build_flow(subset):
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

        return build_flow('training'), build_flow('validation')

    train_gen, validation_gen = flows(data_gen)

    def dataset_from_flow(flow):
        return tf.data.Dataset.from_generator(
            lambda: flow,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, img_h, img_w, 3], [None, num_classes])
        ).repeat()

    train_dataset = dataset_from_flow(train_gen)
    validation_dataset = dataset_from_flow(validation_gen)

    # --------- Training ---------

    model = VGG_transfer_model(img_h, img_w, num_classes, FINETUNING)

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
        steps_per_epoch=len(train_gen),
        validation_data=validation_dataset,
        validation_steps=len(validation_gen),
        callbacks=callbacks_list,
    )


    '''
    562/562 [==============================] - 48s 86ms/step - loss: 0.7169 - accuracy: 0.6569 - val_loss: 0.4852 - val_accuracy: 0.7736
    Epoch 2/100
    562/562 [==============================] - 47s 84ms/step - loss: 0.4637 - accuracy: 0.7963 - val_loss: 0.4945 - val_accuracy: 0.7790
    Epoch 3/100
    562/562 [==============================] - 48s 86ms/step - loss: 0.3469 - accuracy: 0.8544 - val_loss: 0.4232 - val_accuracy: 0.8137
    Epoch 4/100
    562/562 [==============================] - 49s 86ms/step - loss: 0.2461 - accuracy: 0.9025 - val_loss: 0.4587 - val_accuracy: 0.8191
    Epoch 5/100
    562/562 [==============================] - 49s 87ms/step - loss: 0.1542 - accuracy: 0.9450 - val_loss: 0.5308 - val_accuracy: 0.8209
    Epoch 6/100
    562/562 [==============================] - 49s 86ms/step - loss: 0.0935 - accuracy: 0.9659 - val_loss: 0.5706 - val_accuracy: 0.8191
    Epoch 7/100
    562/562 [==============================] - 48s 86ms/step - loss: 0.0706 - accuracy: 0.9744 - val_loss: 0.6243 - val_accuracy: 0.8324
    Epoch 8/100
    562/562 [==============================] - 49s 87ms/step - loss: 0.0329 - accuracy: 0.9907 - val_loss: 0.6994 - val_accuracy: 0.8316
    '''
