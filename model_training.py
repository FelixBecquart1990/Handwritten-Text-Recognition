import tensorflow as tf

from model import flor, ctc_loss_lambda_func, get_callbacks
from prepare_data import *


from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pathlib
# import prepare_data_for_training

import argparse
import os
import datetime



input_size = (1024, 128, 1)
max_text_length = 143
letters = " #'()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstwuvxyzÂÊÔàáâãèéêẹìíòóôõùúýăĐđĩũƠơưạảấầẩẫậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"


# Import json labels file
labels_json_file = './data/labels.json'
labels_json = pd.read_json(labels_json_file, orient='index', encoding="utf-8").reset_index()
labels_json.columns = ['name', 'label']

# Set default variable
max_len = 69
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_link = './data/train'
test_link = './data/test'

train_folder = pathlib.Path(train_link)
test_folder = pathlib.Path(test_link)

all_train_path = [str(item) for item in train_folder.glob('*/') if item.is_file()]
all_train_label = get_label(all_train_path, labels_json, max_len, letters)

train_path, valid_path, train_label, valid_label = train_test_split(all_train_path, all_train_label, test_size = 0.3, random_state = 101)

train_ds = prepare_for_training(train_path, train_label, BATCH_SIZE)
valid_ds = prepare_for_training(valid_path, valid_label, BATCH_SIZE)

TRAIN_SAMPLES = len(train_label)
VALID_SAMPLES = len(valid_label)

num_steps_train = tf.math.ceil(float(TRAIN_SAMPLES)/BATCH_SIZE)   
num_steps_val = tf.math.ceil(float(VALID_SAMPLES)/BATCH_SIZE) 


outs = flor(input_size, 143, 0.001)
inputs, outputs, optimizer = outs
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)
print(model.summary())


checkpoint = './models/checkpoint_1.hdf5'
callback = get_callbacks(checkpoint)


start_time = datetime.datetime.now()

history = model.fit(train_ds,
                    steps_per_epoch = num_steps_train,
                    epochs=5,
                    validation_data = valid_ds,
                    validation_steps = num_steps_val)

total_time = datetime.datetime.now() - start_time

loss = history.history['loss']
val_loss = history.history['val_loss']

min_val_loss = min(val_loss)
min_val_loss_i = val_loss.index(min_val_loss)

time_epoch = (total_time / len(loss))

t_corpus = "\n".join([
    "Batch:                   {}\n".format(BATCH_SIZE),
    "Time per epoch:          {}".format(time_epoch),
    "Total epochs:            {}".format(len(loss)),
    "Best epoch               {}\n".format(min_val_loss_i + 1),
    "Training loss:           {}".format(loss[min_val_loss_i]),
    "Validation loss:         {}".format(min_val_loss),
])

with open(os.path.join("./info/train.txt"), "w") as f:
    f.write(t_corpus)
    print(t_corpus)