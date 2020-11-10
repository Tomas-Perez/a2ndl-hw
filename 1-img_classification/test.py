import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classes import classes
from datetime import datetime

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
img_h = 256
img_w = 256

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

# Print model summary
print(model.summary())

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

callbacks = []

exps_dir = "experiments"
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

model_name = 'MaskDetection'

exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# Model checkpoint
# ----------------
ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
                                                   save_weights_only=True)  # False to save the model directly
callbacks.append(ckpt_callback)

# Early Stopping
# --------------
early_stop = True
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    callbacks.append(es_callback)

model.fit(x=train_dataset,
    epochs=100,
    steps_per_epoch=len(train_gen),
    validation_data=validation_dataset,
    validation_steps=len(validation_gen),
    callbacks=callbacks,
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
