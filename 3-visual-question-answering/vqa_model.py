import os
import numpy as np
from PIL import Image
import random
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import math
from itertools import islice

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set TF logging to ERROR, needs to be done before importing TF
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input 

import callbacks
from labels_dict import labels_dict

MODEL_NAME = 'VQA-model'

def create_model(img_h, img_w, num_classes, max_seq_length, vocabulary_length, embedding_dim, train_mode=True):
    vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))

    if train_mode:
        if FINETUNING:
            freeze_until = 15

            for layer in vgg.layers[:freeze_until]:
                layer.trainable = False
        else:
            vgg.trainable = False

    merge_layer_units = 512

    img_feature_model = tf.keras.Sequential()
    img_feature_model.add(vgg)
    img_feature_model.add(tf.keras.layers.Flatten())
    img_feature_model.add(tf.keras.layers.Dense(units=merge_layer_units, activation='relu'))

    text_feature_model = tf.keras.Sequential()
    text_feature_model.add(tf.keras.Input(shape=[max_seq_length]))
    text_feature_model.add(tf.keras.layers.Embedding(vocabulary_length, embedding_dim, input_length=max_seq_length, mask_zero=True))
    text_feature_model.add(tf.keras.layers.LSTM(units=512, return_sequences=True))
    text_feature_model.add(tf.keras.layers.LSTM(units=512, return_sequences=False))
    text_feature_model.add(tf.keras.layers.Dense(merge_layer_units, activation='relu'))

    final_model_output = tf.keras.layers.Multiply()([img_feature_model.output, text_feature_model.output])
    final_model_output = tf.keras.layers.Dense(units=512, activation='relu')(final_model_output)
    final_model_output = tf.keras.layers.Dense(units=num_classes, activation='softmax')(final_model_output)

    return tf.keras.Model(inputs = [img_feature_model.input, text_feature_model.input], outputs = final_model_output)

def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class VQADataset(tf.keras.utils.Sequence):

    def __init__(self, dataset_dir, which_subset, tokenized_texts, num_classes, img_generator=None, 
        img_preprocessing_function=None, img_out_shape=[256, 256], validation_split=0.2):

        NUMBER_OF_TRAIN_QUESTIONS = 58832
        split_index = math.ceil(NUMBER_OF_TRAIN_QUESTIONS * (1-validation_split))
        
        with open(os.path.join(dataset_dir, 'train_questions_annotations.json')) as f:
            annotations = json.load(f)
            if which_subset == 'training':
                self.annotations = list(islice(annotations.items(), split_index))
            elif which_subset == 'validation':
                self.annotations = list(islice(annotations.items(), split_index, None))
            else:
                raise RuntimeError(f'Unknown subset {which_subset}')

        self.which_subset = which_subset
        self.dataset_dir = dataset_dir
        self.img_generator = img_generator
        self.preprocessing_function = img_preprocessing_function
        self.out_shape = img_out_shape
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        curr_annotation = self.annotations[index][1]
        curr_filename = curr_annotation["image_id"] + ".png"
        curr_question = text_inputs[self.annotations[index][0]]
        correct_idx = labels_dict[curr_annotation["answer"]]
        curr_answer = np.zeros(num_classes)
        curr_answer[correct_idx] = 1
        
        # Read Image and perform augmentation if necessary
        img = Image.open(os.path.join(self.dataset_dir, 'Images', curr_filename)).convert('RGB')
        
        img = img.resize(self.out_shape)
        img_arr = np.array(img)

        if self.which_subset == 'training' and self.img_generator is not None:
            img_t = self.img_generator.get_random_transform(img_arr.shape)
            img_arr = self.img_generator.apply_transform(img_arr, img_t)
        
        if self.preprocessing_function is not None:
            img_arr = self.preprocessing_function(img_arr)

        return (img_arr, curr_question), curr_answer

def build_text_inputs(dataset_dir):
    # Create Tokenizer to convert words to integers
    questions = []
    annotation_ids = []
    with open(os.path.join(dataset_dir, 'train_questions_annotations.json')) as f:
        annotations = json.load(f)
        for a_id, a in annotations.items():
            questions.append(a['question'])
            annotation_ids.append(a_id)

    MAX_NUM_WORDS = 5000
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(questions)
    tokenized = tokenizer.texts_to_sequences(questions)

    word_index = tokenizer.word_index
    max_length = max(len(q) for q in tokenized)
    text_inputs = pad_sequences(tokenized, maxlen=max_length)

    res = dict()
    for i in range(len(annotation_ids)):
        res[annotation_ids[i]] = text_inputs[i]

    return res, len(word_index), max_length

if __name__ == "__main__":
    AUGMENT_DATA = True
    CHECKPOINTS = False
    EARLY_STOP = True
    TENSORBOARD = False
    SAVE_BEST = True
    VALIDATION_SPLIT = 0.15
    FINETUNING = False

    dataset_dir = 'VQA_Dataset'

    text_inputs, num_words, max_seq_length = build_text_inputs(dataset_dir)

    # Set global seed for all internal generators, this should make all randomization reproducible
    import signal
    SEED = signal.SIGSEGV.value # Set SEED to SEG_FAULT code (11)
    set_seeds(SEED)

    img_dim = 128
    
    # Hyper parameters
    bs = 64
    lr = 1e-3
    epochs = 100
    embedding_dim = 32

    num_classes = len(labels_dict)

    model = create_model(img_dim, img_dim, 
        num_classes=num_classes, 
        max_seq_length=max_seq_length, 
        vocabulary_length=num_words, 
        embedding_dim=embedding_dim
    )

    model.summary()

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

    dataset = VQADataset(dataset_dir, 'training', text_inputs, num_classes, 
        img_out_shape=[img_dim, img_dim], 
        validation_split=VALIDATION_SPLIT,
        img_preprocessing_function=preprocess_input,
        img_generator=img_data_gen,
    )
    print(len(dataset))
    dataset_valid = VQADataset(dataset_dir, 'validation', text_inputs, num_classes, 
        img_out_shape=[img_dim, img_dim], 
        validation_split=VALIDATION_SPLIT,
        img_preprocessing_function=preprocess_input,
        img_generator=img_data_gen,
    )
    print(len(dataset_valid))

    train_dataset = tf.data.Dataset.from_generator(
        lambda: dataset,
        output_types=((tf.float32, tf.float32), tf.float32),
        output_shapes=(([img_dim, img_dim, 3], [max_seq_length]), [num_classes]),
    ).batch(bs).repeat()

    valid_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_valid,
        output_types=((tf.float32, tf.float32), tf.float32),
        output_shapes=(([img_dim, img_dim, 3], [max_seq_length]), [num_classes]),
    ).batch(bs).repeat()

    # Loss
    # Sparse Categorical Crossentropy to use integers (mask) instead of one-hot encoded labels
    loss = tf.keras.losses.CategoricalCrossentropy()

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

    now = datetime.now().strftime('%b%d_%H-%M-%S')
    exp_dir = os.path.join(model_dir, str(now))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    callbacks_list = []

    # Model checkpoint
    if CHECKPOINTS:
        callbacks_list.append(callbacks.checkpoints(exp_dir))

    # Early stopping
    if EARLY_STOP:
        callbacks_list.append(callbacks.early_stopping(patience=10))

    # Save best model
    # ----------------
    best_checkpoint_path = None
    if SAVE_BEST:
        best_checkpoint_path, save_best_callback = callbacks.save_best(exp_dir)
        callbacks_list.append(save_best_callback)


    model.fit(
        x=train_dataset,
        epochs=epochs,
        steps_per_epoch=len(dataset) // bs,
        validation_data=valid_dataset,
        validation_steps=len(dataset_valid) // bs,
        callbacks=callbacks_list
    )