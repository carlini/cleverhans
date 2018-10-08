from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

class ScalarReport(object):
  def __init__(self, name, desc, scalar):
    self.name = name
    self.scalar = scalar
    self.desc = desc

  def render(self):
    return "{} : {}".format(self.name, self.scalar)

def sanity_test_accuracy(sess, model, x_test, y_test, X, logits, batch_size):
  acc = 0
  for i in range(0, len(x_test), batch_size):
    pred = np.argmax(sess.run(logits, {X: x_test[i:i+batch_size]}),axis=1)
    label = np.argmax(y_test[i:i+batch_size],axis=1)
    acc += np.sum(pred==label)
  return ScalarReport("Clean Accuracy",
                      "TODO",
                      acc/x_test.shape[0])
    
  

def generate_report(sess, model, x_test, y_test):
  batch_size = 512
  X = tf.placeholder(tf.float32, [None]+list(x_test.shape[1:]))
  Y = tf.placeholder(tf.float32, [None, y_test.shape[1]])
  logits = model.get_logits(X)
  
  
  print("GEN", model)
  print(sanity_test_accuracy(sess, model, x_test, y_test, X, logits, batch_size).render())
  pass
