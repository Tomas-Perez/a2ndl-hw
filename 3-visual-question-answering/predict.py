import os
from datetime import datetime
from labels_dict import labels_dict
from files_in_dir import get_files_in_directory
from vqa_model import create_model, MODEL_NAME
from tokenizerr import get_tokenizer
import numpy as np
from PIL import Image
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set TF logging to ERROR, needs to be done before importing TF
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input 



def create_csv(results, results_dir='./'):
    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')


def build_data(dataset_dir):
    # Create Tokenizer to convert words to integers
    questions = []
    images = []
    annotation_ids = []
    with open(os.path.join(dataset_dir, 'test_questions.json')) as f:
        annotations = json.load(f)
        for a_id, a in annotations.items():
            questions.append(a['question'])
            images.append(a['image_id'] + ".png")
            annotation_ids.append(a_id)
    
    
    MAX_NUM_WORDS = 5000
    tokenizer = get_tokenizer()
    tokenized = tokenizer.texts_to_sequences(questions)
    text_inputs = pad_sequences(tokenized, maxlen=max_seq_length)
    

    '''
    word_index = tokenizer.word_index
    max_length = max(len(q) for q in tokenized)
    
    text_inputs = pad_sequences(tokenized, maxlen=max_length)
    quests = dict()
    imgs   = dict()
    for i in range(len(annotation_ids)):
        quests[annotation_ids[i]] = text_inputs[i]
        imgs[annotation_ids[i]] = images[i]

    return ids, quests, len(word_index), max_length, imgs
    '''
    
    return annotation_ids, text_inputs, images



dataset_dir = 'VQA_Dataset'

num_classes = len(labels_dict)
img_dim = 256
embedding_dim = 32
max_seq_length = 21
num_words = 4640

base_model_exp_dir = f"experiments/{MODEL_NAME}"
saved_weights = [os.path.join(base_model_exp_dir, f) for f in get_files_in_directory(base_model_exp_dir, include_folders=True)]
latest_saved_weights_path = max(saved_weights, key=os.path.getctime)
weights = os.path.join(latest_saved_weights_path, 'best/model')

annotation_ids, text_inputs, images = build_data(dataset_dir)

print(f"{annotation_ids[0]}{text_inputs[0]}{images[0]}")

model = create_model(img_dim, img_dim, 
        num_classes=num_classes, 
        max_seq_length=max_seq_length, 
        vocabulary_length=num_words, 
        embedding_dim=embedding_dim
    )

model.load_weights(weights).expect_partial()

results = {}

for i in range(0, len(annotation_ids)):
    a_id = annotation_ids[i]
    question = np.array(text_inputs[i])
    image = images[i]
    
    img = Image.open(os.path.join(dataset_dir, 'Images', image)).convert('RGB')
    img = img.resize([img_dim, img_dim])
    img_arr = np.array(img)
    img_arr = preprocess_input(img_arr)
    
    print(img_arr.shape)
    
    softmax = model.predict([np.expand_dims(img_arr, 0), np.expand_dims(question, 0)])
    prediction = tf.argmax(softmax, 1)
    prediction = tf.keras.backend.get_value(prediction)[0]
    
    results[a_id] = prediction

create_csv(results)
    








