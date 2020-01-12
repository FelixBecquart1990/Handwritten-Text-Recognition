from model import flor
from tensorflow.keras.models import Model
from model import ctc_loss_lambda_func
import tensorflow as tf
import pandas as pd
import numpy as np
import pathlib
# import prepare_data_for_training

import argparse
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def label_to_text(labels):
    return ''.join(list(map(lambda x: letters[x] if x < len(letters) else "", labels)))

def get_label(all_img_path):
    all_labels = []
    for path in all_img_path:
        img_name = pathlib.Path(path).name
        mask = labels_json['name'] == img_name
        sentence = labels_json['label'][mask].values[0]
        label = text_to_labels(sentence)
        label = pad_sequences([label], maxlen=max_len, padding='post')
        all_labels.append(label)
    return all_labels


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [1024, 128])
    image /= 255.0  # normalize to [0,1] range
    image = 2*image-1  # normalize to [-1,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=100):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
    
input_size = (1024, 128, 1)
max_text_length = 143
letters = " #'()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstwuvxyzÂÊÔàáâãèéêẹìíòóôõùúýăĐđĩũƠơưạảấầẩẫậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"


# Import json labels file
labels_json_file = './data/0916_Data Samples 2/labels.json'
labels_json = pd.read_json(labels_json_file, orient='index', encoding="utf-8").reset_index()
labels_json.columns = ['name', 'label']

TRAIN_SAMPLES = 1823
BATCH_SIZE = 32

num_steps_train = tf.math.ceil(float(TRAIN_SAMPLES)/BATCH_SIZE)   


# Set default variable
max_len = 69

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_folder = './data/train'
raw_folder = pathlib.Path(train_folder)

all_img_path = [str(item) for item in raw_folder.glob('*/') if item.is_file()]
all_img_label = get_label(all_img_path)

train_ds = tf.data.Dataset.from_tensor_slices((all_img_path, all_img_label))
# train_ds = train_ds.map(load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)
# train_ds = prepare_for_training(train_ds, shuffle_buffer_size=100)

outs = flor(input_size, 143, 0.001)

inputs, outputs, optimizer = outs

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)
# model.summary()
# history = model.fit(train_ds,
#                     steps_per_epoch = num_steps_train,
#                     epochs=3)