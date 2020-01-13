import pandas as pd
import argparse
import os
import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np



def text_to_labels(text, letters):
#   return np.asarray(list(map(lambda x: letters.index(x), text)), dtype=np.uint8)
  return list(map(lambda x: letters.index(x), text))

def label_to_text(labels, letters):
    return ''.join(list(map(lambda x: letters[x] if x < len(letters) else "", labels)))

def get_label(all_img_path, labels_json, max_len, letters):
    all_labels = []
    for path in all_img_path:
        img_name = pathlib.Path(path).name
        mask = labels_json['name'] == img_name
        sentence = labels_json['label'][mask].values[0]
        label = text_to_labels(sentence, letters)
        label = pad_sequences([label], maxlen=max_len, padding='post', value = 0)
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

def prepare_for_training(img_path, img_label, batch_size , cache=True, shuffle_buffer_size=100):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    ds = tf.data.Dataset.from_tensor_slices((img_path, img_label))
    ds = ds.map(load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)
    
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds






