import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classes import classes
from datetime import datetime
from PIL import Image

EVAL_ONLY = True
AUGMENT_DATA = False # @Note Worse if augmenting

img_h = 256
img_w = 256
	
# Set global seed for all internal generators, this should make all randomization reproducible
SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)

preprocess_input = tf.keras.applications.vgg16.preprocess_input

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
			      validation_split=0.2,
                              preprocessing_function=preprocess_input)
else:
	data_gen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2, 
                preprocessing_function=preprocess_input)

# Training and validation datasets
dataset_dir = "MaskDataset"

bs = 8

num_classes = len(classes)

training_dir = f"{dataset_dir}/training-structured"

train_gen = data_gen.flow_from_directory(
    training_dir,
    color_mode='rgb',
    target_size=(img_h, img_w),
    batch_size=bs,
    classes=classes,
    class_mode='categorical',
    shuffle=True,
    subset='training',
)

validation_gen = data_gen.flow_from_directory(
    training_dir,
    color_mode='rgb',
    target_size=(img_h, img_w),
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


vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))

vgg.summary()

finetuning = True

if finetuning:
        freeze_until = 15

        for layer in vgg.layers[:freeze_until]:
            layer.trainable = False
else:
    bgg.trainable = False

model = tf.keras.Sequential()
model.add(vgg)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

#print(model.summary())

loss = tf.keras.losses.CategoricalCrossentropy()
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
metrics = ['accuracy']

# Compile Model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

exps_dir = "transfer_experiments"
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

model_name = 'MaskDetection'

exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

callbacks = []

# Model checkpoint
ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), save_weights_only=True)  # False to save the model directly
callbacks.append(ckpt_callback)


# Early stopping
early_stop = True
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    callbacks.append(es_callback)


# load weights or fit
if(EVAL_ONLY):
    #model.load_weights('transfer_experiments/MaskDetection_Nov10_15-41-51/ckpts/cp_08.ckpt')  # use this if you want to restore saved model
    model.load_weights('transfer_experiments/MaskDetection_Nov10_19-42-19/ckpts/cp_08.ckpt')
else:
    model.fit(x=train_dataset,
        epochs=100,
        steps_per_epoch=len(train_gen),
        validation_data=validation_dataset,
        validation_steps=len(validation_gen),
        callbacks=callbacks,
    )


# -------- CSV output --------
def create_csv(results, results_dir='./results'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')


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
    #img = make_square(img)
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


'''
# Test
test_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_dir = f"{dataset_dir}/test"

test_gen = test_data_gen.flow_from_directory(test_dir,
                                             batch_size=bs,
                                             class_mode='categorical',
                                             shuffle=False,
                                             seed=SEED)

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, img_h, img_w, 3], [None, num_classes])
).repeat()


eval_out = model.evaluate(x=test_dataset,
                          steps=len(test_gen),
                          verbose=0)

print(eval_out)
'''


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
