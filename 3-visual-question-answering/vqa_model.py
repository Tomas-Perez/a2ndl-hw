import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set TF logging to ERROR, needs to be done before importing TF
import numpy as np
import random
from datetime import datetime
import math
from tokens import build_text_inputs
from glove import get_embeddings
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input 
import tensorflow.keras.backend as K
import callbacks
from labels_dict import labels_dict
from dataset import VQADataset

MODEL_NAME = 'VQA-model'

def create_model(img_h, img_w, num_classes, max_seq_length, embedding_matrix, train_mode=True):
    merge_layer_units = 512

    img_feature_model = tf.keras.Sequential()
    vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))
    vgg.trainable = False
    img_feature_model.add(vgg)
    img_feature_model.add(tf.keras.layers.Flatten())
    img_feature_model.add(tf.keras.layers.Dense(units=merge_layer_units))
    img_feature_model.add(tf.keras.layers.BatchNormalization())
    img_feature_model.add(tf.keras.layers.ReLU())

    text_feature_model = tf.keras.Sequential()
    text_feature_model.add(tf.keras.Input(shape=[max_seq_length]))
    embedding_layer = tf.keras.layers.Embedding(
        embedding_matrix.shape[0], 
        embedding_matrix.shape[1], 
        weights=[embedding_matrix], 
        input_length=max_seq_length, 
        mask_zero=True
    )
    embedding_layer.trainable = False
    text_feature_model.add(embedding_layer)
    text_feature_model.add(tf.keras.layers.LSTM(units=512, return_sequences=True))
    text_feature_model.add(tf.keras.layers.LSTM(units=512, return_sequences=False))
    text_feature_model.add(tf.keras.layers.Dense(merge_layer_units))
    text_feature_model.add(tf.keras.layers.BatchNormalization())
    text_feature_model.add(tf.keras.layers.ReLU())

    final_model_output = tf.keras.layers.Multiply()([img_feature_model.output, text_feature_model.output])
    attention = tf.keras.layers.Attention()([text_feature_model.output, final_model_output])
    final_model_output = tf.keras.layers.Concatenate()([final_model_output, attention])
    final_model_output = tf.keras.layers.BatchNormalization()(final_model_output)
    final_model_output = tf.keras.layers.Dense(units=merge_layer_units, activation='relu')(final_model_output)
    final_model_output = tf.keras.layers.Dense(units=num_classes, activation='softmax')(final_model_output)

    return tf.keras.Model(inputs = [img_feature_model.input, text_feature_model.input], outputs = final_model_output)

def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    AUGMENT_DATA = True
    EARLY_STOP = True
    SAVE_BEST = True
    VALIDATION_SPLIT = 0.15

    num_k = math.ceil(1 / VALIDATION_SPLIT)

    max_seq_length = 21

    dataset_dir = 'VQA_Dataset'

    text_inputs = build_text_inputs(dataset_dir, max_seq_length)

    # Set global seed for all internal generators, this should make all randomization reproducible
    import signal
    SEED = signal.SIGSEGV.value # Set SEED to SEG_FAULT code (11)
    set_seeds(SEED)

    img_dim = 256
    
    # Hyper parameters
    bs = 64
    lr = 1e-3
    epochs = 100

    num_classes = len(labels_dict)

    if AUGMENT_DATA:
        img_data_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=20,
            height_shift_range=20,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode= 'reflect',
        )
    else:
        img_data_gen = None

    now = datetime.now().strftime('%b%d_%H-%M-%S')

    for i in range(num_k):

        model = create_model(img_dim, img_dim, 
            num_classes=num_classes, 
            max_seq_length=max_seq_length, 
            embedding_matrix=get_embeddings(),
        )

        if i == 0:
            model.summary()

        print(f"Training for k index={i}/{num_k}")

        dataset = VQADataset(dataset_dir, 'training', text_inputs, num_classes, 
            img_out_shape=[img_dim, img_dim], 
            validation_split=VALIDATION_SPLIT,
            img_preprocessing_function=preprocess_input,
            img_generator=img_data_gen,
            k_idx=i,
        )
        dataset_valid = VQADataset(dataset_dir, 'validation', text_inputs, num_classes, 
            img_out_shape=[img_dim, img_dim], 
            validation_split=VALIDATION_SPLIT,
            img_preprocessing_function=preprocess_input,
            img_generator=img_data_gen,
            k_idx=i,
        )

        train_dataset = tf.data.Dataset.from_generator(
            lambda: dataset,
            output_types=((tf.float32, tf.float32), tf.int32),
            output_shapes=(([img_dim, img_dim, 3], [max_seq_length]), []),
        ).batch(bs).repeat()

        valid_dataset = tf.data.Dataset.from_generator(
            lambda: dataset_valid,
            output_types=((tf.float32, tf.float32), tf.int32),
            output_shapes=(([img_dim, img_dim, 3], [max_seq_length]), []),
        ).batch(bs).repeat()

        # Loss
        # Regular SparseCategoricalCrossentropy proved to be the best performing
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        # learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Validation metrics
        metrics = ['accuracy']

        # Compile Model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # ---- Callbacks ----
        exps_dir = "experiments"
        if not os.path.exists(exps_dir):
            os.makedirs(exps_dir)

        model_dir = os.path.join(exps_dir, MODEL_NAME)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        exp_dir = os.path.join(model_dir, str(now))
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        current_k_idx_dir = os.path.join(exp_dir, f"k_{i}")
        if not os.path.exists(current_k_idx_dir):
            os.makedirs(current_k_idx_dir)

        callbacks_list = []

        # Early stopping
        if EARLY_STOP:
            callbacks_list.append(callbacks.early_stopping(patience=10))

        # Save best model
        # ----------------
        best_checkpoint_path = None
        if SAVE_BEST:
            best_checkpoint_path, save_best_callback = callbacks.save_best(current_k_idx_dir)
            callbacks_list.append(save_best_callback)

        model.fit(
            x=train_dataset,
            epochs=epochs,
            steps_per_epoch=len(dataset) // bs,
            validation_data=valid_dataset,
            validation_steps=len(dataset_valid) // bs,
            callbacks=callbacks_list
        )

        # Clear tensorflow session to release memory, otherwise it keeps rising after each fold
        K.clear_session()
