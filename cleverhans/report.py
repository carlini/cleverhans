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
  """
  Generate a report for the model accuracy on the clean data
  """
  acc = get_accuracy(sess, X, logits, x_test, y_test, batch_size)
  return ScalarReport("Clean Accuracy",
                      "TODO",
                      acc)
    
def report_for_attack(sess, model, x_test, y_test, X, logits, batch_size,
                      attack, attack_kwargs):
  """
  Generate a report for a specific attack and set of attack hyperparameters.
  
  This function is likely only useful for quick testing. It is almost always
  better to generate reports at multiple settings of hyperparameters,
  especially if the distortion bound is one of those parameters.
  """

  adv = attack.generate_np(x_test, y=y_test, **attack_kwargs)

  print(x_test.shape, adv.shape)
  acc = get_accuracy(sess, X, logits, adv, y_test, batch_size)
  return ScalarReport("Adv Accuracy",
                      "TODO",
                      acc)

def report_sweep_epsilon(sess, model, x_test, y_test, X, logits, batch_size,
                         attack, eps_max, granularity, attack_kwargs,
                         eps_iter_fn):
  """
  Generate a report for a specific attack at one set of hyperparameters but
  varrying the epsilon distortion bound.

  Instead of performing a naive sweep of all epsilon values, uses binary
  search on a per-example basis to determine the minimal distortion.
  """

  
  final_adv = []
  for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]
    last_worked = np.copy(x_batch)
    epsilon = np.ones((len(x_batch),1,1,1))*(eps_max/2)

    os = np.zeros(500, dtype=np.bool)
    adjust_epsilon = eps_max/4
    while adjust_epsilon*4 >= granularity:
      attack_kwargs['eps'] = epsilon
      # For a fixed (epsilon, nb_iter), compute the eps_iter
      attack_kwargs['eps_iter'] = eps_iter_fn(epsilon, attack_kwargs['nb_iter'])

      adv = attack.generate_np(x_batch, y=y_batch, **attack_kwargs)

      preds = np.argmax(sess.run(logits, {X: adv}), axis=1)

      attack_success = preds != np.argmax(y_batch, axis=1)
      
      epsilon[np.logical_not(attack_success)] += adjust_epsilon
      epsilon[attack_success] -= adjust_epsilon
      os |= attack_success
      last_worked[attack_success] = adv[attack_success]
      adjust_epsilon /= 2
    print(np.mean(os))
    print(np.mean(epsilon))
    final_adv.extend(last_worked)

  delta = np.max(np.abs(final_adv-x_test),axis=(1,2,3))

  return GraphReport("Adversarial Epsilon Sweep",
                     "TODO",
                     delta)

def report_sweep_iterations(sess, defended_model, 
                            x_test, y_test, X,
                            defended_logits, batch_size,
                            iterations_min, iterations_max, granularity,
                            attack, attack_kwargs, eps_iter_fn):
  """
  Generate a report exploring the tradeoff from using more iterations of
  the attack at a fixed distortion bound.
  """

  outer_success_rate = []
  for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]

    success_rate = []
    for iterations in range(iterations_min, iterations_max, granularity):
      attack_kwargs['nb_iter'] = iterations
      # For a fixed (epsilon, nb_iter), compute the eps_iter
      attack_kwargs['eps_iter'] = eps_iter_fn(attack_kwargs['eps'], iterations)

      adv = attack.generate_np(x_batch, y=y_batch, **attack_kwargs)

      preds = np.argmax(sess.run(defended_logits, {X: adv}), axis=1)

      attack_success = preds != np.argmax(y_batch, axis=1)

      success_rate.append(np.mean(attack_success))
    print(success_rate)
    outer_success_rate.append(success_rate)
  print(outer_success_rate)
  outer_success_rate = np.mean(outer_success_rate, axis=0)
  print(outer_success_rate)


def transfer_from_undefended(sess, defended_model, undefended_model,
                             x_test, y_test, X,
                             defended_logits, undefended_logits, batch_size,
                             eps_max, granularity, attack):
  """
  Generate a report that attemps a transferability test for a specific attack.
  """
  raise 
  attack_success = []
  transfer_success = []
  for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]
    for epsilon in np.arange(0,eps_max,granularity):

      adv = attack.generate_np(x_batch, y=y_batch,
                               nb_iter=30,
                               eps=epsilon,
                               eps_iter=epsilon)
      
      preds = np.argmax(sess.run(undefended_logits, {X: adv}), axis=1)
      batch_attack_success = preds != np.argmax(y_batch, axis=1)

      preds_defended = np.argmax(sess.run(defended_logits, {X: adv}), axis=1)
      batch_transfer_success = preds_defended != np.argmax(y_batch, axis=1)

      attack_success += np.sum(batch_attack_success)
      transfer_success += np.sum(batch_transfer_success)
  return GraphReport("Transfer success epsilon sweep",
                     "TODO",
                     attack_success/len(x_test),
                     transfer_success/len(x_test))


def generate_report(sess, defended_model, x_test, y_test,
                    undefended_models=[]):
  batch_size = 500
  X = tf.placeholder(tf.float32, [None]+list(x_test.shape[1:]))
  Y = tf.placeholder(tf.float32, [None, y_test.shape[1]])
  defended_logits = defended_model.get_logits(X)
  undefended_logits = [model.get_logits(X) for model in undefended_models]
  
  
  print(report_accuracy(sess, defended_model, x_test, y_test, X, defended_logits, batch_size).render())
  #print(report_accuracy(sess, undefended_model, x_test, y_test, X, undefended_logits, batch_size).render())

  eps_iter_fn = lambda eps, steps: eps/(steps**.5)
  
  print(report_sweep_iterations(sess, defended_model, 
                                x_test, y_test, X,
                                defended_logits, batch_size,
                                1, 100, 5,
                                cleverhans.attacks.MadryEtAl(defended_model, sess=sess),
                                {'eps': .1}, eps_iter_fn))
  #print(transfer_from_undefended(sess, defended_model, undefended_model,
  #                               x_test, y_test, X,
  #                               defended_logits, undefended_logits, batch_size, .9, 0.1).render())
  
  #print(report_sweep_epsilon(sess, model, x_test, y_test, X, logits, batch_size,
  #                           cleverhans.attacks.MadryEtAl(model, sess=sess), 0.3, 0.001,
  #                           {'nb_iter': 30}).render())
  pass
