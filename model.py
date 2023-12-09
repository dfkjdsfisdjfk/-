from tensorflow import keras
from keras import layers
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
import tensorflow as tf
from keras.layers import Layer


class prelayer:
  def __init__(self):
    self = self

  def __call__(self, layer):
    # layer = keras.layers.experimental.preprocessing.Normalization(mean=(0.485, 0.456, 0.406), variance=(0.229 ** 2, 0.224 ** 2, 0.225 ** 2))(layer)

    layer = keras.layers.ZeroPadding2D(padding=(3, 3))(layer)
    layer = keras.layers.Conv2D(64, (3, 3), strides=(2, 2))(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Activation('relu')(layer)
    layer = keras.layers.ZeroPadding2D(padding=(1, 1))(layer)

    layer = keras.layers.MaxPooling2D((3, 3), 2)(layer)

    return layer

class ResidualBlock(tf.keras.layers.Layer):
  def __init__(self, n_filter, n_filter2, number):
    super(ResidualBlock, self).__init__()
    self.n_filter = n_filter
    self.n_filter2 = n_filter2
    self.number = number

  def __call__(self, layer):
    shortcut = layer

    layer = keras.layers.Conv2D(self.n_filter, (1, 1), strides=(1, 1), padding='valid', name = f"{self.number}_{self.n_filter}_{self.n_filter2}conv")(layer)
    layer = keras.layers.BatchNormalization(name = f"{self.number}BatchNormal")(layer)
    layer = keras.layers.Activation('relu', name = f"{self.number}Active")(layer)
    layer = keras.layers.Conv2D(self.n_filter, (3, 3), strides=(1, 1), padding='same', name = f"{self.number}_{self.n_filter}_{self.n_filter2}conv2")(layer)
    layer = keras.layers.BatchNormalization(name = f"{self.number}BatchNormal2")(layer)
    layer = keras.layers.Activation('relu', name = f"{self.number}Active2")(layer)
    layer = keras.layers.Conv2D(self.n_filter2, (1, 1), strides=(1, 1), padding='valid', name = f"{self.number}_{self.n_filter}_{self.n_filter2}conv3")(layer)
    layer = keras.layers.BatchNormalization(name = f"{self.number}BatchNormal3")(layer)
    shortcut = keras.layers.Conv2D(self.n_filter2, (1, 1), strides=(1, 1), padding='valid', name = f"{self.number}_{self.n_filter}_{self.n_filter2}conv4")(shortcut)            
    shortcut = keras.layers.BatchNormalization(name = f"{self.number}BatchNormal4")(shortcut)
    layer = keras.layers.Add()([layer, shortcut])
    layer = keras.layers.Activation('relu', name = f"{self.number}Active3")(layer)

    return layer

class resnet(tf.keras.Model):
  def __init__(self, num_classes=20):
    self.num_classes = num_classes

  def get_layer(self, input_layer):
    layer = prelayer()(input_layer)
    layer = ResidualBlock(64, 256, 1)(layer)
    layer = ResidualBlock(64, 256, 2)(layer)
    layer = ResidualBlock(64, 256, 3)(layer) 

    layer = keras.layers.MaxPooling2D((3, 3), 2)(layer)

    layer = ResidualBlock(128, 512, 4)(layer)
    layer = ResidualBlock(128, 512, 5)(layer)
    layer = ResidualBlock(128, 512, 6)(layer)

    layer = keras.layers.MaxPooling2D((3, 3), 2)(layer)

    layer = ResidualBlock(256, 1024, 7)(layer)
    layer = ResidualBlock(256, 1024, 8)(layer)
    layer = ResidualBlock(256, 1024, 9)(layer)

    layer = keras.layers.MaxPooling2D((3, 3), 2)(layer)

    layer = ResidualBlock(512, 2048, 10)(layer)
    layer = ResidualBlock(512, 2048, 11)(layer)
    layer = ResidualBlock(512, 2048, 12)(layer)

    layer = keras.layers.GlobalAveragePooling2D()(layer)
    layer = Dense(20, activation='softmax')(layer)

    return layer

  def build_model(self, num_classes=20):

    input_layer = keras.layers.Input(shape = (224,224,3))
    layer = self.get_layer(input_layer)

    model = keras.models.Model(input_layer, layer)

    return model

def get_classifier(num_classes=20):
  return resnet().build_model(20)