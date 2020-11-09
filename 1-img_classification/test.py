import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classes import classes
from sklearn.model_selection import train_test_split
	
SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Get current working directory
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

dataset_dir = "MaskDataset"

bs = 8
img_h = 256
img_w = 256

num_classes = len(classes)

# Training
training_dir = f"{dataset_dir}/training-structured"

train_gen = data_gen.flow_from_directory(
    training_dir,
    batch_size=bs,
    classes=classes,
    class_mode='categorical',
    shuffle=True,
    subset='training',
)

validation_gen = data_gen.flow_from_directory(
    training_dir,
    batch_size=bs,
    classes=classes,
    class_mode='categorical',
    shuffle=True,
    subset='validation',
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, None, None, 3], [None, num_classes])
).repeat()

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, None, None, 3], [None, num_classes])
).repeat()

# Architecture: Features extraction -> Classifier

start_f = 8
depth = 5

model = tf.keras.Sequential()

# Features extraction
for i in range(depth):

    if i == 0:
        input_shape = [None, None, 3]
    else:
        input_shape=[None]

    # Conv block: Conv2D -> Activation -> Pooling
    model.add(tf.keras.layers.Conv2D(filters=start_f, 
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    start_f *= 2
    
# Classifier
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))


# Optimization params
# -------------------

# Loss
loss = tf.keras.losses.CategoricalCrossentropy()

# learning rate
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']
# ------------------

# Compile Model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.fit(x=train_dataset,
    epochs=10,  #### set repeat in training dataset
    steps_per_epoch=len(train_gen),
    validation_data=validation_dataset,
    validation_steps=len(validation_gen), 
)