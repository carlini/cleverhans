from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.train import train
from cleverhans_tutorials.tutorial_models import ModelBasicCNN, make_basic_picklable_cnn
from cleverhans.report import generate_report
from cleverhans.model import CallableModelWrapper

import keras

FLAGS = flags.FLAGS

BATCH_SIZE = 128
NB_EPOCHS = 6
SOURCE_SAMPLES = 10
LEARNING_RATE = .001
ATTACK_ITERATIONS = 100
TARGETED = True

def make_model(sess, x_train, y_train, rest=""):
  model = keras.models.Sequential()
  layers = [keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                input_shape=(28,28,1)),
            keras.layers.MaxPool2D(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPool2D(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(128,activation='relu'),
            keras.layers.Dense(10,activation='softmax')]
  for layer in layers:
    model.add(layer)
  model.summary()

  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='adam')

  if os.path.exists("models/mnist.model"):
    model.load_weights("models/mnist.model")
  else:
    model.fit(x_train,
              y_train, epochs=5, batch_size=128)
    model.save_weights("models/mnist.model")
  return CallableModelWrapper(lambda x: tf.log(model(x)), 'logits')

def make_autoencoder(sess, x_train):
  model = keras.models.Sequential()
  layers = [keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                                input_shape=(28,28,1)),
            keras.layers.MaxPool2D(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPool2D(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.UpSampling2D(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.UpSampling2D(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')]
  for layer in layers:
    model.add(layer)

  model.compile(loss=keras.losses.mean_squared_error,
                optimizer='adam')

  if os.path.exists("models/autoencoder.model"):
    model.load_weights("models/autoencoder.model")
  else:
    for i in range(2):
      model.fit(x_train+np.random.normal(0, .2, size=x_train.shape),
                x_train, epochs=1, batch_size=128)
    model.save_weights("models/autoencoder.model")
  return model

def almost_binarize(model):
  def fn(x):
    return model.get_logits(1/(1+tf.exp(-100*(x-.5))))
  return CallableModelWrapper(fn, 'logits')

def add_noise(model):
  def fn(x):
    return model.get_logits(x+tf.random_normal(tf.shape(x), 0, .2))
  return CallableModelWrapper(fn, 'logits')

def cleaner(dae, model):
  def fn(x):
    #xx = dae(dae(dae(dae(dae(dae(x))))))
    xx = dae(x)
    return model.get_logits(xx)
  return CallableModelWrapper(fn, 'logits')

if __name__ == "__main__":
  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Create TF session
  sess = tf.Session()
  keras.backend.set_session(sess)
  print("Created TensorFlow session.")

  set_log_level(logging.DEBUG)

  # Get MNIST test data
  mnist = MNIST()
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Obtain Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  #defended_model = make_model(sess, x_train, y_train, "_other")
  undefended_model = make_model(sess, x_train, y_train, "_baselineq")

  dae = make_autoencoder(sess, x_train)
  defended_model = cleaner(dae, undefended_model)
  
  generate_report(sess, defended_model, x_test, y_test,
                  dataset_name="mnist",
                  undefended_models=[undefended_model])
    
