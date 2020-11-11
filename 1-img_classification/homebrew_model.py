import os
import random
import numpy as np
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set TF logging to ERROR, needs to be done before importing TF
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# local imports
from classes import classes
import callbacks

MODEL_NAME = "MaskDetection-Homebrew"

def homebrew_model(img_h, img_w, num_classes):
    # --------- Training ---------
    # Architecture: Features extraction -> Classifier

    start_f = 8
    depth = 6

    model = tf.keras.Sequential()

    # Features extraction
    for i in range(depth):

        if i == 0:
            input_shape = [img_h, img_w, 3]
        else:
            input_shape=[None]

        # Conv block: Conv2D -> Activation -> Pooling
        # @Note tried different initializers, default is working best so far
        model.add(tf.keras.layers.Conv2D(
            filters=start_f, 
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            input_shape=input_shape
        ))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        start_f *= 2
        
    # Classifier
    # model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    # Optimization params
    # -------------------

    # Loss
    loss = tf.keras.losses.CategoricalCrossentropy()

    # learning rate
    lr = 1e-4 # @Note tried different ones, this one is the best
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # -------------------

    # Validation metrics
    # ------------------

    metrics = ['accuracy']
    # ------------------

    # Compile Model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    AUGMENT_DATA = True # @Note Worse if augmenting
    CHECKPOINTS = False
    EARLY_STOP = True
    TENSORBOARD = False
    SAVE_BEST = True
	
    # Set global seed for all internal generators, this should make all randomization reproducible
    import signal
    SEED = signal.SIGSEGV.value
    set_seeds(SEED)

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
            validation_split=0.2
        )
    else:
        data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

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

    model = homebrew_model(img_h, img_w, num_classes)

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

    # Model checkpoint
    # ----------------
    if CHECKPOINTS:
        callbacks_list.append(callbacks.checkpoints())

    # Save best model
    # ----------------
    if SAVE_BEST:
        callbacks_list.append(callbacks.save_best(exp_dir))

    # Early Stopping
    # --------------
    if EARLY_STOP:
        callbacks_list.append(callbacks.early_stopping(patience=5))

    # Visualize Learning on Tensorboard
    # ---------------------------------
    if TENSORBOARD:
        callbacks_list.append(callbacks.tensorboard(exp_dir))

    # Train model
    # ---------------------------------
    model.fit(
        x=train_dataset,
        epochs=100,
        steps_per_epoch=len(train_gen),
        validation_data=validation_dataset,
        validation_steps=len(validation_gen),
        callbacks=callbacks_list,
    )

    """
    Best so far (see best.txt for full logs)

    Epoch 1/100
    562/562 [==============================] - 24s 43ms/step - loss: 1.0989 - accuracy: 0.3451 - val_loss: 1.0992 - val_accuracy: 0.3396
    Epoch 2/100
    562/562 [==============================] - 23s 42ms/step - loss: 1.0965 - accuracy: 0.3631 - val_loss: 1.0948 - val_accuracy: 0.3636

    Epoch 17/100
    562/562 [==============================] - 23s 41ms/step - loss: 0.3627 - accuracy: 0.8493 - val_loss: 0.7742 - val_accuracy: 0.6684
    Epoch 18/100
    562/562 [==============================] - 23s 41ms/step - loss: 0.3241 - accuracy: 0.8651 - val_loss: 0.7710 - val_accuracy: 0.6622

    """
