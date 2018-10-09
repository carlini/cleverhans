from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cleverhans.attacks

class Report(object):
  def __init__(self):
    pass

  def render(self):
    return

  def show(self):
    return

class ScalarReport(Report):
  def __init__(self, name, desc, scalar):
    self.name = name
    self.scalar = scalar
    self.desc = desc

  def render(self):
    return "{} : {}".format(self.name, self.scalar)

class GraphReport(Report):
  def __init__(self, name, desc, xs, ys):
    self.name = name
    self.xs = xs
    self.ys = ys
    self.desc = desc

  def render(self):
    plt.plot(self.xs, self.ys)
    plt.show()

def get_accuracy(sess, X, logits, xs, ys, batch_size):
  acc = 0
  for i in range(0, len(xs), batch_size):
    pred = np.argmax(sess.run(logits, {X: xs[i:i+batch_size]}),axis=1)
    label = np.argmax(ys[i:i+batch_size],axis=1)
    acc += np.sum(pred==label)
  return acc/xs.shape[0]
    
def report_accuracy(sess, model, x_test, y_test, X, logits, batch_size):
  acc = get_accuracy(sess, X, logits, x_test, y_test, batch_size)
  return ScalarReport("Clean Accuracy",
                      "TODO",
                      acc)
    
def report_at_epsilon(sess, model, x_test, y_test, X, logits, batch_size,
                      attack, attack_kwargs):

  adv = attack.generate_np(x_test, y=y_test, **attack_kwargs)

  print(x_test.shape, adv.shape)
  acc = get_accuracy(sess, X, logits, adv, y_test, batch_size)
  return ScalarReport("Adv Accuracy",
                      "TODO",
                      acc)

def report_sweep_epsilon(sess, model, x_test, y_test, X, logits, batch_size,
                         attack, eps_max, granularity, attack_kwargs):
  
  #"""
  final_adv = []
  for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]
    last_worked = np.copy(x_batch)
    epsilon = np.ones((len(x_batch),1,1,1))*(eps_max/2)

    os = np.zeros(500, dtype=np.bool)
    adjust_epsilon = eps_max/4
    while adjust_epsilon*4 >= granularity:
      attack_kwargs['eps_iter'] = epsilon*.1
      attack_kwargs['eps'] = epsilon
      attack.feed = x_batch
      adv = attack.generate_np(x_batch, y=y_batch, **attack_kwargs)

      preds = np.argmax(sess.run(logits, {X: adv}), axis=1)
      #print(epsilon.flatten())
      #print(preds == np.argmax(y_batch, axis=1))

      attack_success = preds != np.argmax(y_batch, axis=1)
      
      epsilon[np.logical_not(attack_success)] += adjust_epsilon
      epsilon[attack_success] -= adjust_epsilon
      os |= attack_success
      last_worked[attack_success] = adv[attack_success]
      adjust_epsilon /= 2
    print(np.mean(os))
    print(np.mean(epsilon))
    final_adv.extend(last_worked)
  np.save("/tmp/a.npy", final_adv)
  #"""
  final_adv = np.load("/tmp/a.npy")
  delta = np.max(np.abs(final_adv-x_test),axis=(1,2,3))

  plt.hist(delta,100)
  plt.show()

def transfer_from_undefended(sess, defended_model, undefended_model,
                             x_test, y_test, X,
                             defended_logits, undefended_logits, batch_size,
                             eps_max, granularity):

  attack = cleverhans.attacks.MadryEtAl(undefended_model, sess=sess)
  for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]

    for epsilon in np.arange(0,eps_max,granularity):
      adv = attack.generate_np(x_batch, y=y_batch,
                               nb_iter=30,
                               eps=epsilon,
                               eps_iter=epsilon/10)
      
      preds = np.argmax(sess.run(undefended_logits, {X: adv}), axis=1)
      attack_success = preds != np.argmax(y_batch, axis=1)

      preds_defended = np.argmax(sess.run(defended_logits, {X: adv}), axis=1)
      transfer_success = preds_defended != np.argmax(y_batch, axis=1)

      print(epsilon, np.mean(attack_success), np.mean(transfer_success))
  

def generate_report(sess, defended_model, undefended_model, x_test, y_test):
  batch_size = 500
  X = tf.placeholder(tf.float32, [None]+list(x_test.shape[1:]))
  Y = tf.placeholder(tf.float32, [None, y_test.shape[1]])
  defended_logits = defended_model.get_logits(X)
  undefended_logits = undefended_model.get_logits(X)
  
  
  print(report_accuracy(sess, defended_model, x_test, y_test, X, defended_logits, batch_size).render())
  print(report_accuracy(sess, undefended_model, x_test, y_test, X, undefended_logits, batch_size).render())
  exit(0)
  print(transfer_from_undefended(sess, defended_model, undefended_model,
                                 x_test, y_test, X,
                                 defended_logits, undefended_logits, batch_size, .9, 0.1).render())
  
  #print(report_sweep_epsilon(sess, model, x_test, y_test, X, logits, batch_size,
  #                           cleverhans.attacks.MadryEtAl(model, sess=sess), 0.3, 0.001,
  #                           {'nb_iter': 30}).render())
  pass
