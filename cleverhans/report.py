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
  def __init__(self, name, desc, xs, ys, x_label=None,
               y_label=None, ys_05=None, ys_95=None):
    self.name = name
    self.xs = xs
    self.ys = ys
    self.ys_05 = ys_05 or ys
    self.ys_95 = ys_95 or ys
    self.desc = desc
    self.x_label
    self.y_label
    

  def render(self):
    plt.plot(self.xs, self.ys)
    plt.show()

class Data(object):
  pass

class SummaryData(Data):
  pass

    
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

def batched_report(fn):
  def wrap(_sentinel=None, **kwargs):
    if _sentinel is not None:
      raise Exception("You must call report functions using only keyword arguments.")
    batch_size = kwargs['batch_size'] or 100
    x_test = kwargs['x_test']
    y_test = kwargs['y_test']

    del kwargs['batch_size']
    del kwargs['x_test']
    del kwargs['y_test']
    
    outer_result = []
    
    for i in range(0, len(x_test), batch_size):
      kwargs['x_test'] = x_test[i:i+batch_size]
      kwargs['y_test'] = y_test[i:i+batch_size]

      batch_result = fn(**kwargs)
      print(batch_result)
      outer_result.extend(batch_result)
    return outer_result
      
  return wrap

@batched_report
def report_sweep_epsilon(sess, defended_model, x_test, y_test,
                         X, defended_logits,
                         attack, eps_max, granularity, attack_kwargs,
                         eps_iter_fn):
  """
  Generate a report for a specific attack at one set of hyperparameters but
  varrying the epsilon distortion bound.

  Instead of performing a naive sweep of all epsilon values, uses binary
  search on a per-example basis to determine the minimal distortion.
  """

  
  last_worked = np.copy(x_test)
  epsilon = np.ones((len(x_test),1,1,1))*(eps_max/2)

  os = np.zeros(len(x_test), dtype=np.bool)
  adjust_epsilon = eps_max/4
  while adjust_epsilon*4 >= granularity:
    attack_kwargs['eps'] = epsilon
    # For a fixed (epsilon, nb_iter), compute the eps_iter
    attack_kwargs['eps_iter'] = eps_iter_fn(epsilon, attack_kwargs['nb_iter'])

    adv = attack.generate_np(x_test, y=y_test, **attack_kwargs)

    preds = np.argmax(sess.run(defended_logits, {X: adv}), axis=1)

    attack_success = preds != np.argmax(y_test, axis=1)
    
    epsilon[np.logical_not(attack_success)] += adjust_epsilon
    epsilon[attack_success] -= adjust_epsilon
    os |= attack_success
    last_worked[attack_success] = adv[attack_success]
    adjust_epsilon /= 2

  delta = np.max(np.abs(last_worked-x_test),axis=(1,2,3))

  return delta

@batched_report
def report_sweep_iterations(sess, defended_model, 
                            x_test, y_test, X,
                            defended_logits,
                            iterations_min, iterations_max, granularity,
                            attack, attack_kwargs, eps_iter_fn):
  """
  Generate a report exploring the tradeoff from using more iterations of
  the attack at a fixed distortion bound.
  """

  success_rate = []
  for iterations in range(iterations_min, iterations_max, granularity):
    attack_kwargs['nb_iter'] = iterations
    # For a fixed (epsilon, nb_iter), compute the eps_iter
    attack_kwargs['eps_iter'] = eps_iter_fn(attack_kwargs['eps'], iterations)

    adv = attack.generate_np(x_test, y=y_test, **attack_kwargs)

    preds = np.argmax(sess.run(defended_logits, {X: adv}), axis=1)

    attack_success = preds != np.argmax(y_test, axis=1)

    success_rate.append(attack_success)
    
  return np.array(success_rate).T

@batched_report
def transfer_from_undefended(sess, defended_model, undefended_model,
                             x_test, y_test, X,
                             defended_logits, undefended_logits,
                             eps_max, granularity, attack):
  """
  Generate a report that attemps a transferability test for a specific attack.
  """
  attack_success = []
  transfer_success = []
  for epsilon in np.arange(0,eps_max,granularity):

    adv = attack.generate_np(x_batch, y=y_batch,
                             nb_iter=30,
                             eps=epsilon,
                             eps_iter=epsilon)
    
    preds = np.argmax(sess.run(undefended_logits, {X: adv}), axis=1)
    batch_attack_success = preds != np.argmax(y_batch, axis=1)

    preds_defended = np.argmax(sess.run(defended_logits, {X: adv}), axis=1)
    batch_transfer_success = preds_defended != np.argmax(y_batch, axis=1)

    attack_success.append(batch_attack_success)
    transfer_success.append(batch_transfer_success)

  return np.array([attack_success, transfer_success]).T


class Message:
  def __init__(self, severity, message):
    self.severity = severity
    self.message = message

def perform_sanity_checks(results, datset_name):
  messages = []
  for result in results:
    if result.threatmodel == 'white_box' and isinstance(result,GraphReport):
      if result.x_label in ['iterations', 'epsilon']:
        # If this is a plot of success rate vs (iterations|epsilon) then it should be monotonically
        # increasing.
        for i in range(len(result.xs)):
          if result.ys_95[i] < np.min(result.ys_05[:i]):
            # the 95th percentile at this point is BELOW the 5th percentile of a prior
            # run, which means something is almost certainly broken.
            messages.append(Message("ERROR",
                                    "White-box " + result.x_label + "-sweep evaluation flawed: decreases "
                                    " at index " + result.xs[i] + "."))
      if result.x_label == "epsilon":
        if result.ys_95[-1] != 0:
          messages.append(Message("WARN",
                                  "White-box epsilon-sweep evaluation does not eventually "
                                  "reach 0% model accuracy (100% attack success)."))
      if result.x_label == "iterations":
        if result.ys_05[-1] > result.ys_95[len(result.ys_95)//2]:
          messages.append(Message("WARN",
                                  "White-box iteration-sweep evaluation does not converge:"
                                  "the attack succes is still increasing."))
        
    elif result.threatmodel == 'black_box' and isinstance(result,GraphReport):
      if result.x_label == 'epsilon':
        pass

  if dataset_name in ["mnist", "cifar-10", "imagenet"]:
    if dataset_name == "mnist":
      messages.append(Message("WARN",
                              "MNIST results very often do not transfer to any other dataset. "
                              "Any conclusions drawn from this report are likely to be incorrect "
                              "if not combined with results from a more complicated dataset."))

    # These are thresholds on a per-dataset basis that are "too good to be true"
    # for various severity levels.
    # - SOTA means that it exceeds the state of the art by an order of magnitude,
    #   and while this may be a true result, deserves special attention.
    # - WARN means that the result is probably wrong, either because humans
    #   can not even solve the dataset at that distortion, or because it is so
    #   clearly outside of the bounds of what can be done today that something
    #   is almost certainly going wrong.
    # - ERROR means that the attack must have failed. It is impossible, on this
    #   dataset, to reach the given accuracy numbers.
    thresholds = {
      'mnist': {
        'l_infinity_sota': [],
        'l_infinity_warn': [(0.5, 0.0)],
        'l_infinity_error': [(0.5, 0.1)],
        'l_2_warn': [(6, 0.0)],
        'l_2_error': [(9, 0.0)]
      },
      'cifar': {
        'l_infinity_sota': [(0.03, 0.9), (0.06, 0.5)],
        'l_infinity_warn': [(0.06, 0.9), (0.2, 0.5), (.4, .1)],
        'l_infinity_error': [(0.5, 0.1)],
        'l_2_warn': [(4, 0.1)],
        'l_2_error': [(12, 0.1)]
      },
      'imagenet': {
        'l_infinity_sota': [(0.03, 0.5), (0.06, 0.3)],
        'l_infinity_warn': [(0.06, 0.9)],
        'l_infinity_error': [(0.5, 0.001)],
        'l_2_sota': [(5, 0.001)],
        'l_2_warn': [(10, 0.001)],
        'l_2_error': [(60, 0.001)]
      }}
            
      

def generate_report(sess, defended_model,
                    x_test, y_test,
                    dataset_name=None,
                    undefended_models=[]):
  batch_size = 500
  X = tf.placeholder(tf.float32, [None]+list(x_test.shape[1:]))
  Y = tf.placeholder(tf.float32, [None, y_test.shape[1]])
  defended_logits = defended_model.get_logits(X)
  undefended_logits = [model.get_logits(X) for model in undefended_models]
  
  
  print(report_accuracy(sess, defended_model, x_test, y_test, X, defended_logits, batch_size).render())
  #print(report_accuracy(sess, undefended_model, x_test, y_test, X, undefended_logits, batch_size).render())

  eps_iter_fn = lambda eps, steps: eps/(steps**.5)

  """
  print(report_sweep_iterations(sess=sess,
                                defended_model=defended_model, 
                                x_test=x_test,
                                y_test=y_test,
                                X=X,
                                defended_logits=defended_logits,
                                batch_size=batch_size,
                                iterations_min=1,
                                iterations_max=30,
                                granularity=5,
                                attack=cleverhans.attacks.MadryEtAl(defended_model, sess=sess),
                                attack_kwargs={'eps': .1},
                                eps_iter_fn=eps_iter_fn))
  """

  print(report_sweep_epsilon(sess=sess,
                                defended_model=defended_model, 
                                x_test=x_test[:1000],
                                y_test=y_test[:1000],
                                X=X,
                                defended_logits=defended_logits,
                                batch_size=batch_size,
                                eps_max=0.8,
                                granularity=.01,
                                attack=cleverhans.attacks.MadryEtAl(defended_model, sess=sess),
                                attack_kwargs={'nb_iter': 10},
                                eps_iter_fn=eps_iter_fn))

  
  #print(transfer_from_undefended(sess, defended_model, undefended_model,
  #                               x_test, y_test, X,
  #                               defended_logits, undefended_logits, batch_size, .9, 0.1).render())
  
  #print(report_sweep_epsilon(sess, model, x_test, y_test, X, logits, batch_size,
  #                           cleverhans.attacks.MadryEtAl(model, sess=sess), 0.3, 0.001,
  #                           {'nb_iter': 30}).render())
  pass
