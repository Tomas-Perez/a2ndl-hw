import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classes import classes

AUGMENT_DATA = False # @Note Worse if augmenting
	
SEED = 1234

# Set global seed for all internal generators, this should make all randomization reproducible
SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Data generator
# Single image generator, splitting the entire dataset in training and validation
if AUGMENT_DATA:
	data_gen = ImageDataGenerator(rotation_range=10,
		              width_shift_range=10,
		              height_shift_range=10,
		              zoom_range=0.3,
		              horizontal_flip=True,
		              vertical_flip=True,
		              fill_mode='constant',
		              cval=0,
			      rescale=1./255, 
			      validation_split=0.2)
else:
	data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training and validation datasets
dataset_dir = "MaskDataset"

bs = 8

num_classes = len(classes)

training_dir = f"{dataset_dir}/training-structured"

train_gen = data_gen.flow_from_directory(
    training_dir,
    color_mode='rgb',
    batch_size=bs,
    classes=classes,
    class_mode='categorical',
    shuffle=True,
    subset='training',
)

validation_gen = data_gen.flow_from_directory(
    training_dir,
    color_mode='rgb',
    batch_size=bs,
    classes=classes,
    class_mode='categorical',
    shuffle=True,
    subset='validation',
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, img_h, img_w, 3], [None, num_classes])
).repeat()

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, img_h, img_w, 3], [None, num_classes])
).repeat()



# --------- Training ---------
# Architecture: Features extraction -> Classifier

start_f = 12 #8
depth = 5

model = tf.keras.Sequential()

# Features extraction
for i in range(depth):

    if i == 0:
        input_shape = [img_h, img_w, 3]
    else:
        input_shape=[None]

    # Conv block: Conv2D -> Activation -> Pooling
    # @Note tried different initializers, default is working best so far
    model.add(tf.keras.layers.Conv2D(filters=start_f, 
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='valid',
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
lr = 1e-4 # @Note tried different ones, this one is the best
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


''' Best so far 
Epoch 1/10
562/562 [==============================] - 178s 317ms/step - loss: 1.0994 - accuracy: 0.3362 - val_loss: 1.0982 - val_accuracy: 0.3387
Epoch 2/10
562/562 [==============================] - 177s 315ms/step - loss: 1.0982 - accuracy: 0.3417 - val_loss: 1.0974 - val_accuracy: 0.3645
Epoch 3/10
562/562 [==============================] - 180s 321ms/step - loss: 1.0967 - accuracy: 0.3528 - val_loss: 1.0975 - val_accuracy: 0.3512
Epoch 4/10
562/562 [==============================] - 181s 322ms/step - loss: 1.0946 - accuracy: 0.3691 - val_loss: 1.0976 - val_accuracy: 0.3449
Epoch 5/10
562/562 [==============================] - 181s 322ms/step - loss: 1.0951 - accuracy: 0.3655 - val_loss: 1.0958 - val_accuracy: 0.3645
Epoch 6/10
562/562 [==============================] - 182s 325ms/step - loss: 1.0926 - accuracy: 0.3713 - val_loss: 1.0938 - val_accuracy: 0.3547
Epoch 7/10
562/562 [==============================] - 179s 319ms/step - loss: 1.0910 - accuracy: 0.3713 - val_loss: 1.0940 - val_accuracy: 0.3770
Epoch 8/10
562/562 [==============================] - 180s 321ms/step - loss: 1.0876 - accuracy: 0.3911 - val_loss: 1.0837 - val_accuracy: 0.3788
Epoch 9/10
562/562 [==============================] - 175s 311ms/step - loss: 1.0720 - accuracy: 0.4118 - val_loss: 1.0537 - val_accuracy: 0.4733
Epoch 10/10
562/562 [==============================] - 167s 297ms/step - loss: 1.0337 - accuracy: 0.4675 - val_loss: 0.9951 - val_accuracy: 0.5036
'''
