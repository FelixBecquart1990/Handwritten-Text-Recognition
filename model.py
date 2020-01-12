from tensorflow.keras.preprocessing.sequence import pad_sequences
# import our model, different layers and activation function 
from tensorflow.keras.layers import Layer, Multiply, Activation, Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, PReLU, GRU, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.constraints import MaxNorm
import os
import tensorflow as tf

class FullGatedConv2D(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        """Apply gated convolution"""

        output = super(FullGatedConv2D, self).call(inputs)
        linear = Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, self.nb_filters:])

        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""

        output_shape = super(FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        """Return the config of the layer"""

        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config

def flor(input_size, d_model, learning_rate):
    """
    Gated Convolucional Recurrent Neural Network by Flor et al.
    """

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=40, kernel_size=(2,4), strides=(2,4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(3,3), padding="same", kernel_constraint=MaxNorm(4, [0,1,2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(3,3), padding="same", kernel_constraint=MaxNorm(4, [0,1,2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=56, kernel_size=(2,4), strides=(2,4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(3,3), padding="same", kernel_constraint=MaxNorm(4, [0,1,2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid")(cnn)

    shape = cnn.get_shape()
    nb_units = shape[2] * shape[3]

    bgru = Reshape((shape[1], nb_units))(cnn)

    bgru = Bidirectional(GRU(units=nb_units, return_sequences=True, dropout=0.5))(bgru)
    bgru = Dense(units=nb_units * 2)(bgru)

    bgru = Bidirectional(GRU(units=nb_units, return_sequences=True, dropout=0.5))(bgru)
    output_data = Dense(units=d_model, activation="softmax")(bgru)

    if learning_rate is None:
        learning_rate = 5e-4

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    return (input_data, output_data, optimizer)

def ctc_loss_lambda_func(y_true, y_pred):
    """Function for computing the CTC loss"""

    if len(y_true.shape) > 2:
        y_true = tf.squeeze(y_true)

    # y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
    # output of every model is softmax
    # so sum across alphabet_size_1_hot_encoded give 1
    #               string_length give string length
    input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
    input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

    # y_true strings are padded with 0
    # so sum of non-zero gives number of characters in this string
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    # average loss across all entries in the batch
    loss = tf.reduce_mean(loss)

    return loss