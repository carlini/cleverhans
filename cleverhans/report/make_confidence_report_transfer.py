"""
make_confidence_report_transfer.py
Usage:
  python make_confidence_report.py target_model.joblib source_model.joblib

  where model.joblib is a file created by cleverhans.serial.save containing
  a picklable cleverhans.model.Model instance.

This script will run the model on a variety of types of data and save a
report to model_report.joblib. The report can be later loaded by another
script using cleverhans.serial.load. The format of the report is a dictionary.
Each dictionary key is the name of a type of data:
  clean : Clean data
  semantic : Semantic adversarial examples
  mc: MaxConfidence adversarial examples
Each value in the dictionary contains an array of bools indicating whether
the model got each example correct and an array containing the confidence
that the model assigned to each prediction.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
import math
import six
import time

from cleverhans.evaluation import correctness_and_confidence
from cleverhans.attacks import MaxConfidence
from cleverhans.utils import set_log_level
from cleverhans.serial import load, save

FLAGS = flags.FLAGS

from cleverhans.utils_tf import infer_devices
devices = infer_devices()
num_devices = len(devices)
BATCH_SIZE = 128 * num_devices
TRAIN_START = 0
TRAIN_END = 60000
TEST_START = 0
TEST_END = 10000
WHICH_SET = 'test'
MC_BATCH_SIZE = 16 * num_devices
REPORT_PATH = None
NB_ITER = 40
BASE_EPS_ITER = None # Differs by dataset



from cleverhans.attacks import Semantic


def make_confidence_report(target_filepath, source_filepath, train_start=TRAIN_START, train_end=TRAIN_END,
                           test_start=TEST_START, test_end=TEST_END,
                           batch_size=BATCH_SIZE, which_set=WHICH_SET,
                           mc_batch_size=MC_BATCH_SIZE,
                           report_path=REPORT_PATH):
  """
  Load a saved model, gather its predictions, and save a confidence report.
  :param filepath: path to model to evaluate
  :param train_start: index of first training set example to use
  :param train_end: index of last training set example to use
  :param test_start: index of first test set example to use
  :param test_end: index of last test set example to use
  :param batch_size: size of evaluation batches
  :param mc_batch_size: batch size for MaxConfidence attack
  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.INFO)

  # Create TF session
  sess = tf.Session()

  if report_path is None:
    assert target_filepath.endswith('.joblib')
    report_path = target_filepath[:-len('.joblib')] + "_transfer_report.joblib"

  with sess.as_default():
    model = load(target_filepath)
    assert len(model.get_params()) > 0
    source_model = load(source_filepath)
    assert len(model.get_params()) > 0
    assert len(source_model.get_params()) > 0.
  factory = model.dataset_factory
  factory.kwargs['test_start'] = test_start
  factory.kwargs['test_end'] = test_end
  dataset = factory()
  img_rows, img_cols, nchannels = dataset.x_train.shape[1:4]
  nb_classes = dataset.NB_CLASSES

  center = factory.kwargs['center']
  value_range = 1. + center
  min_value = 0. - center

  if 'CIFAR' in str(factory.cls):
    base_eps = 8. / 255.
    if FLAGS.base_eps_iter is None:
      base_eps_iter = 2. / 255.
    else:
      base_eps_iter = FLAGS.base_eps_iter
  elif 'MNIST' in str(factory.cls):
    base_eps = .3
    if FLAGS.base_eps_iter is None:
      base_eps_iter = .1
    else:
      base_eps_iter= FLAGS.base_eps_iter
  else:
    raise NotImplementedError(str(factory.cls))

  mc_params = {'eps': base_eps * value_range,
               'eps_iter': base_eps_iter * value_range,
               'nb_iter': FLAGS.nb_iter,
               'clip_min': min_value,
               'clip_max': 1.}


  x_data, y_data = dataset.get_set(which_set)

  report = {}

  mc = MaxConfidence(source_model, sess=sess)

  jobs = [('clean', None, None, None),
          ('mc', mc, mc_params, mc_batch_size)]


  for job in jobs:
    name, attack, attack_params, job_batch_size = job
    if job_batch_size is None:
      job_batch_size = batch_size
    t1 = time.time()
    packed = correctness_and_confidence(sess, model, x_data, y_data,
                                        batch_size=job_batch_size, devices=devices,
                                        attack=attack,
                                        attack_params=attack_params)
    t2 = time.time()
    print("Evaluation took", t2 - t1, "seconds")
    correctness, confidence = packed

    report[name] = {
      'correctness' : correctness,
      'confidence'  : confidence
    }

    print_stats(correctness, confidence, name)


  save(report_path, report)


def main(argv=None):
  """
  Make a confidence report and save it to disk.
  """
  try:
    name_of_script, target_filepath, source_filepath = argv
  except ValueError:
    raise ValueError(argv)
  make_confidence_report(target_filepath=target_filepath, source_filepath=source_filepath,
                         test_start=FLAGS.test_start,
                         test_end=FLAGS.test_end, which_set=FLAGS.which_set,
                         report_path=FLAGS.report_path)

def print_stats(correctness, confidence, name):
  """
  Prints out accuracy, coverage, etc. statistics
  :param correctness: ndarray
    One bool per example specifying whether it was correctly classified
  :param confidence: ndarray
    The probability associated with each prediction
  :param name: str
    The name of this type of data (e.g. "clean", "MaxConfidence")
  """
  accuracy = correctness.mean()
  wrongness = 1 - correctness
  denom1 = np.maximum(1, wrongness.sum())
  ave_prob_on_mistake = (wrongness * confidence).sum() / denom1
  assert ave_prob_on_mistake <= 1., ave_prob_on_mistake
  denom2 = np.maximum(1, correctness.sum())
  ave_prob_on_correct = (correctness * confidence).sum() / denom2
  covered = confidence > 0.5
  cov_half = covered.mean()
  acc_half = (correctness * covered).sum() / np.maximum(1, covered.sum())
  print('Accuracy on %s examples: %0.4f' % (name, accuracy))
  print("Average prob on mistakes: %0.4f" % ave_prob_on_mistake)
  print("Average prob on correct: %0.4f" % ave_prob_on_correct)
  print("Accuracy when prob thresholded at .5: %0.4f" % acc_half)
  print("Coverage when prob thresholded at .5: %0.4f" % cov_half)

  success_rate = acc_half * cov_half
  # Success is correctly classifying a covered example
  print("Success rate at .5: %0.4f" % success_rate)
  # Failure is misclassifying a covered example
  failure_rate = (1. - acc_half) * cov_half
  print("Failure rate at .5: %0.4f" % failure_rate)
  print()

if __name__ == '__main__':
  flags.DEFINE_integer('train_start', TRAIN_START, 'Starting point (inclusive)'
                       'of range of train examples to use')
  flags.DEFINE_integer('train_end', TRAIN_END, 'Ending point (non-inclusive) '
                       'of range of train examples to use')
  flags.DEFINE_integer('test_start', TEST_START, 'Starting point (inclusive) of range'
                       ' of test examples to use')
  flags.DEFINE_integer('test_end', TEST_END, 'End point (non-inclusive) of range'
                       ' of test examples to use')
  flags.DEFINE_integer('nb_iter', NB_ITER, 'Number of iterations of PGD')
  flags.DEFINE_string('which_set', WHICH_SET, '"train" or "test"')
  flags.DEFINE_string('report_path', REPORT_PATH, 'Path to save to')
  flags.DEFINE_integer('mc_batch_size', MC_BATCH_SIZE,
                       'Batch size for MaxConfidence')
  flags.DEFINE_float('base_eps_iter', BASE_EPS_ITER, 'epsilon per iteration, if data were in [0, 1]')
  tf.app.run()
