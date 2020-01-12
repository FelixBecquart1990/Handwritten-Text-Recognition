
import pandas as pd
import argparse
import os
import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Get images paths from origine folder
# all_img_path = [str(item) for item in img_train_folder.glob('**/*.*') if item.is_file()]
# len_img_path = len(all_img_path)
# print(len_img_path, " (length of len_img_path)")
# print(all_img_path[:5])

def text_to_labels(text):
  for c in text:
    if letters.find(c) == -1:
      print(c)
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


# Import json labels file
labels_json_file = './data/0916_Data Samples 2/labels.json'
labels_json = pd.read_json(labels_json_file, orient='index').reset_index()
labels_json.columns = ['name', 'label']

# Set default variable
max_len = 100 
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
letters = " #'()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuwvxyzÂÊÔàáâãèẹéêìíòóôõùúýăĐđĩũƠơưạảấầẫẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"

train_folder = './data/train'
raw_folder = pathlib.Path(train_folder)
all_img_path = [str(item) for item in raw_folder.glob('*/') if item.is_file()]

all_img_labels = get_label(all_img_path)
print(all_img_labels[:5])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default='oui', type=str)
    args = parser.parse_args()
    
    
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    # tester(args.test)
    # images_preprocessing_and_saving(data_path, all_img_path)
    # all_img_labels = get_label(all_img_path)
