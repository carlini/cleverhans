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
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from cleverhans.report import generate_report

FLAGS = flags.FLAGS

BATCH_SIZE = 128
NB_EPOCHS = 6
SOURCE_SAMPLES = 10
LEARNING_RATE = .001
ATTACK_ITERATIONS = 100
TARGETED = True

def make_model(sess, x_train, y_train, rest=""):
  # Define TF model graph
  before_vars = set(x.name for x in tf.trainable_variables())
  model = ModelBasicCNN('model'+rest, nb_classes, 64)
  model_vars = [x for x in tf.trainable_variables() if x.name not in before_vars]
  loss = CrossEntropy(model, smoothing=0.1)
  print("Defined TensorFlow model graph.")

  model_path = "models/mnist"+rest
  # Train an MNIST model
  train_params = {
      'nb_epochs': 10,
      'batch_size': BATCH_SIZE,
      'learning_rate': LEARNING_RATE,
      'filename': os.path.split(model_path)[-1]
  }

  rng = np.random.RandomState([2017, 8, 30])

  # check if we've trained before, and if we have, use that pre-trained model
  saver = tf.train.Saver(var_list=model_vars)
  if os.path.exists(model_path + ".meta"):
    saver.restore(sess, model_path)
  else:
    train(sess, loss, x_train, y_train,
          args=train_params, rng=rng,
          var_list=model_vars)
    saver.save(sess, model_path)
  return model

if __name__ == "__main__":
  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Create TF session
  sess = tf.Session()
  print("Created TensorFlow session.")

  set_log_level(logging.DEBUG)

  # Get MNIST test data
  mnist = MNIST()
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Obtain Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  defended_model = make_model(sess, x_train, y_train, "_baseline")
  undefended_model = make_model(sess, x_train, y_train, "_other")
  generate_report(sess, defended_model, undefended_model, x_test, y_test)
    
