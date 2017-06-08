from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
from six.moves import xrange
import tensorflow as tf
import warnings

from . import utils_tf
from . import utils

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def fgsm(x, predictions, eps=0.3, clip_min=None, clip_max=None):
    return fgm(x, predictions, y=None, eps=eps, ord=np.inf, clip_min=clip_min,
               clip_max=clip_max)


def fgm(x, preds, y=None, eps=0.3, ord=np.inf, clip_min=None, clip_max=None):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor
    :param y: (optional) A placeholder for the model labels. Only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :return: a tensor for the adversarial example
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        signed_grad = tf.sign(grad)
    elif ord == 1:
        reduc_ind = list(xrange(1, len(x.get_shape())))
        signed_grad = grad / tf.reduce_sum(tf.abs(grad),
                                           reduction_indices=reduc_ind,
                                           keep_dims=True)
    elif ord == 2:
        reduc_ind = list(xrange(1, len(x.get_shape())))
        signed_grad = grad / tf.sqrt(tf.reduce_sum(tf.square(grad),
                                                   reduction_indices=reduc_ind,
                                                   keep_dims=True))
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


def vatm(model, x, logits, eps, num_iterations=1, xi=1e-6,
         clip_min=None, clip_max=None, scope=None):
    """
    Tensorflow implementation of the perturbation method used for virtual
    adversarial training: https://arxiv.org/abs/1507.00677
    :param model: the model which returns the network unnormalized logits
    :param x: the input placeholder
    :param logits: the model's unnormalized output tensor
    :param eps: the epsilon (input variation parameter)
    :param num_iterations: the number of iterations
    :param xi: the finite difference parameter
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :param seed: the seed for random generator
    :return: a tensor for the adversarial example
    """
    with tf.name_scope(scope, "virtual_adversarial_perturbation"):
        d = tf.random_normal(tf.shape(x))
        for i in range(num_iterations):
            d = xi * utils_tf.l2_batch_normalize(d)
            logits_d = model(x + d)
            kl = utils_tf.kl_with_logits(logits, logits_d)
            Hd = tf.gradients(kl, d)[0]
            d = tf.stop_gradient(Hd)
        d = eps * utils_tf.l2_batch_normalize(d)
        adv_x = tf.stop_gradient(x + d)
        if (clip_min is not None) and (clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
        return adv_x


def apply_perturbations(i, j, X, increase, theta, clip_min, clip_max):
    """
    TensorFlow implementation for apply perturbations to input features based
    on salency maps
    :param i: index of first selected feature
    :param j: index of second selected feature
    :param X: a matrix containing our input features for our sample
    :param increase: boolean; true if we are increasing pixels, false otherwise
    :param theta: delta for each feature adjustment
    :param clip_min: mininum value for a feature in our sample
    :param clip_max: maximum value for a feature in our sample
    : return: a perturbed input feature matrix for a target class
    """

    # perturb our input sample
    if increase:
        X[0, i] = np.minimum(clip_max, X[0, i] + theta)
        X[0, j] = np.minimum(clip_max, X[0, j] + theta)
    else:
        X[0, i] = np.maximum(clip_min, X[0, i] - theta)
        X[0, j] = np.maximum(clip_min, X[0, j] - theta)

    return X


def saliency_map(grads_target, grads_other, search_domain, increase):
    """
    TensorFlow implementation for computing saliency maps
    :param grads_target: a matrix containing forward derivatives for the
                         target class
    :param grads_other: a matrix where every element is the sum of forward
                        derivatives over all non-target classes at that index
    :param search_domain: the set of input indices that we are considering
    :param increase: boolean; true if we are increasing pixels, false otherwise
    :return: (i, j, search_domain) the two input indices selected and the
             updated search domain
    """
    # Compute the size of the input (the number of features)
    nf = len(grads_target)

    # Remove the already-used input features from the search space
    invalid = list(set(range(nf)) - search_domain)
    increase_coef = (2 * int(increase) - 1)
    grads_target[invalid] = - increase_coef * np.max(np.abs(grads_target))
    grads_other[invalid] = increase_coef * np.max(np.abs(grads_other))

    # Create a 2D numpy array of the sum of grads_target and grads_other
    target_sum = grads_target.reshape((1, nf)) + grads_target.reshape((nf, 1))
    other_sum = grads_other.reshape((1, nf)) + grads_other.reshape((nf, 1))

    # Create a mask to only keep features that match saliency map conditions
    if increase:
        scores_mask = ((target_sum > 0) & (other_sum < 0))
    else:
        scores_mask = ((target_sum < 0) & (other_sum > 0))

    # Create a 2D numpy array of the scores for each pair of candidate features
    scores = scores_mask * (-target_sum * other_sum)

    # A pixel can only be selected (and changed) once
    np.fill_diagonal(scores, 0)

    # Extract the best two pixels
    best = np.argmax(scores)
    p1, p2 = best % nf, best // nf

    # Remove used pixels from our search domain
    search_domain.discard(p1)
    search_domain.discard(p2)

    return p1, p2, search_domain


def jacobian(sess, x, grads, target, X, nb_features, nb_classes):
    """
    TensorFlow implementation of the foward derivative / Jacobian
    :param x: the input placeholder
    :param grads: the list of TF gradients returned by jacobian_graph()
    :param target: the target misclassification class
    :param X: numpy array with sample input
    :param nb_features: the number of features in the input
    :return: matrix of forward derivatives flattened into vectors
    """
    # Prepare feeding dictionary for all gradient computations
    feed_dict = {x: X}

    # Initialize a numpy array to hold the Jacobian component values
    jacobian_val = np.zeros((nb_classes, nb_features), dtype=np.float32)

    # Compute the gradients for all classes
    for class_ind, grad in enumerate(grads):
        run_grad = sess.run(grad, feed_dict)
        jacobian_val[class_ind] = np.reshape(run_grad, (1, nb_features))

    # Sum over all classes different from the target class to prepare for
    # saliency map computation in the next step of the attack
    other_classes = utils.other_classes(nb_classes, target)
    grad_others = np.sum(jacobian_val[other_classes, :], axis=0)

    return jacobian_val[target], grad_others


def jacobian_graph(predictions, x, nb_classes):
    """
    Create the Jacobian graph to be ran later in a TF session
    :param predictions: the model's symbolic output (linear output,
        pre-softmax)
    :param x: the input placeholder
    :param nb_classes: the number of classes the model has
    :return:
    """
    # This function will return a list of TF gradients
    list_derivatives = []

    # Define the TF graph elements to compute our derivatives for each class
    for class_ind in xrange(nb_classes):
        derivatives, = tf.gradients(predictions[:, class_ind], x)
        list_derivatives.append(derivatives)

    return list_derivatives


def jsma(sess, x, predictions, grads, sample, target, theta, gamma, clip_min,
         clip_max):
    """
    TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
    for details about the algorithm design choices).
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output (linear output,
        pre-softmax)
    :param grads: symbolic gradients
    :param sample: numpy array with sample input
    :param target: target class for sample input
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param clip_min: minimum value for components of the example returned
    :param clip_max: maximum value for components of the example returned
    :return: an adversarial sample
    """

    # Copy the source sample and define the maximum number of features
    # (i.e. the maximum number of iterations) that we may perturb
    adv_x = copy.copy(sample)
    # count the number of features. For MNIST, 1x28x28 = 784; for
    # CIFAR, 3x32x32 = 3072; etc.
    nb_features = np.product(adv_x.shape[1:])
    # reshape sample for sake of standardization
    original_shape = adv_x.shape
    adv_x = np.reshape(adv_x, (1, nb_features))
    # compute maximum number of iterations
    max_iters = np.floor(nb_features * gamma / 2)

    # Find number of classes based on grads
    nb_classes = len(grads)

    increase = bool(theta > 0)

    # Compute our initial search domain. We optimize the initial search domain
    # by removing all features that are already at their maximum values (if
    # increasing input features---otherwise, at their minimum value).
    if increase:
        search_domain = set([i for i in xrange(nb_features)
                             if adv_x[0, i] < clip_max])
    else:
        search_domain = set([i for i in xrange(nb_features)
                             if adv_x[0, i] > clip_min])

    # Initialize the loop variables
    iteration = 0
    adv_x_original_shape = np.reshape(adv_x, original_shape)
    current = utils_tf.model_argmax(sess, x, predictions, adv_x_original_shape)

    # Repeat this main loop until we have achieved misclassification
    while (current != target and iteration < max_iters and
           len(search_domain) > 1):
        # Reshape the adversarial example
        adv_x_original_shape = np.reshape(adv_x, original_shape)

        # Compute the Jacobian components
        grads_target, grads_others = jacobian(sess, x, grads, target,
                                              adv_x_original_shape,
                                              nb_features, nb_classes)

        # Compute the saliency map for each of our target classes
        # and return the two best candidate features for perturbation
        i, j, search_domain = saliency_map(
            grads_target, grads_others, search_domain, increase)

        # Apply the perturbation to the two input features selected previously
        adv_x = apply_perturbations(
            i, j, adv_x, increase, theta, clip_min, clip_max)

        # Update our current prediction by querying the model
        current = utils_tf.model_argmax(sess, x, predictions,
                                        adv_x_original_shape)

        # Update loop variables
        iteration = iteration + 1

    # Compute the ratio of pixels perturbed by the algorithm
    percent_perturbed = float(iteration * 2) / nb_features

    # Report success when the adversarial example is misclassified in the
    # target class
    if current == target:
        return np.reshape(adv_x, original_shape), 1, percent_perturbed
    else:
        return np.reshape(adv_x, original_shape), 0, percent_perturbed


def jsma_batch(sess, x, pred, grads, X, theta, gamma, clip_min, clip_max,
               nb_classes, targets=None):
    """
    Applies the JSMA to a batch of inputs
    :param sess: TF session
    :param x: the input placeholder
    :param pred: the model's symbolic output
    :param grads: symbolic gradients
    :param X: numpy array with sample inputs
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param clip_min: minimum value for components of the example returned
    :param clip_max: maximum value for components of the example returned
    :param nb_classes: number of model output classes
    :param targets: target class for sample input
    :return: adversarial examples
    """
    X_adv = np.zeros(X.shape)

    for ind, val in enumerate(X):
        val = np.expand_dims(val, axis=0)
        if targets is None:
            # No targets provided, randomly choose from other classes
            from .utils_tf import model_argmax
            gt = model_argmax(sess, x, pred, val)

            # Randomly choose from the incorrect classes for each sample
            from .utils import random_targets
            target = random_targets(gt, nb_classes)[0]
        else:
            target = targets[ind]

        X_adv[ind], _, _ = jsma(sess, x, pred, grads, val, np.argmax(target),
                                theta, gamma, clip_min, clip_max)

    return np.asarray(X_adv, dtype=np.float32)


def jacobian_augmentation(sess, x, X_sub_prev, Y_sub, grads, lmbda,
                          keras_phase=None):
    """
    Augment an adversary's substitute training set using the Jacobian
    of a substitute model to generate new synthetic inputs.
    See https://arxiv.org/abs/1602.02697 for more details.
    See tutorials/mnist_blackbox.py for example use case
    :param sess: TF session in which the substitute model is defined
    :param x: input TF placeholder for the substitute model
    :param X_sub_prev: substitute training data available to the adversary
                       at the previous iteration
    :param Y_sub: substitute training labels available to the adversary
                  at the previous iteration
    :param grads: Jacobian symbolic graph for the substitute
                  (should be generated using attacks_tf.jacobian_graph)
    :param keras_phase: (deprecated) if not None, holds keras learning_phase
    :return: augmented substitute data (will need to be labeled by oracle)
    """
    assert len(x.get_shape()) == len(np.shape(X_sub_prev))
    assert len(grads) >= np.max(Y_sub) + 1
    assert len(X_sub_prev) == len(Y_sub)

    if keras_phase is not None:
        warnings.warn("keras_phase argument is deprecated and will be removed"
                      " on 2017-09-28. Instead, use K.set_learning_phase(0) at"
                      " the start of your script and serve with tensorflow.")

    # Prepare input_shape (outside loop) for feeding dictionary below
    input_shape = list(x.get_shape())
    input_shape[0] = 1

    # Create new numpy array for adversary training data
    # with twice as many components on the first dimension.
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, input in enumerate(X_sub_prev):
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Prepare feeding dictionary
        feed_dict = {x: np.reshape(input, input_shape)}

        # Compute sign matrix
        grad_val = sess.run([tf.sign(grad)], feed_dict=feed_dict)[0]

        # Create new synthetic point in adversary substitute training set
        X_sub[2*ind] = X_sub[ind] + lmbda * grad_val

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub


class CarliniWagnerL2:
    def __init__(self, sess, model, batch_size, confidence,
                 targeted, learning_rate,
                 binary_search_steps, max_iterations,
                 abort_early, initial_const,
                 clip_min, clip_max, num_labels, shape):
        """
        This attack was originally proposed by Carlini and Wagner. It is an
        iterative attack that finds adversarial examples on many defenses that
        are robust to other attacks.
        Paper link: https://arxiv.org/abs/1608.04644

        :param confidence: Confidence of adversarial examples: higher produces
                           examples that are farther away, but more strongly
                           classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param targeted: True if we should perform a targetted attack, False
                         otherwise.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between distance and confidence.
        :param max_iterations: The maximum number of iterations. Larger values
                               are more accurate; setting too small will
                               require a large learning rate and will produce
                               poor results.
        :param abort_early: If true, allows early aborts if gradient descent
                            gets stuck.
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of distance and confidence.
                              If binary_search_steps is large, the initial
                              constant is not important.
        :param clip_min: Minimum input component value
        :param clip_max: Maximum input component value
        :param num_labels: The number of different classes for the classifier
        :param shape: The shape of the input tensor, without the number of
                      batches
        """

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = model

        self.repeat = binary_search_steps >= 10

        self.shape = shape = tuple([batch_size]+list(shape))

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32,
                                name='timg')
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)),
                                dtype=tf.float32, name='tlab')
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32,
                                 name='const')

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape,
                                          name='assign_timg')
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size, num_labels),
                                          name='assign_tlab')
        self.assign_const = tf.placeholder(tf.float32, [batch_size],
                                           name='assign_const')

        # the resulting image, tanh'd to keep bounded from clip_min
        # to clip_max
        self.newimg = (tf.tanh(modifier + self.timg)+1)/2
        self.newimg = self.newimg*(clip_max-clip_min)+clip_min

        # prediction BEFORE-SOFTMAX of the model
        self.output = model(self.newimg)

        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg -
                                              (tf.tanh(self.timg) + 1) / 2 *
                                              (clip_max-clip_min)+clip_min),
                                    list(range(1, len(shape))))

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab) * self.output, 1)
        other = tf.reduce_max((1 - self.tlab) * self.output - self.tlab*10000,
                              1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real-other+self.CONFIDENCE)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        self.loss = self.loss1+self.loss2

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels
        If self.targeted is false, then targets are the original class labels
        """

        r = []
        for i in range(0, len(imgs), self.batch_size):
            r.extend(self.attack_batch(imgs[i:i+self.batch_size],
                                       targets[i:i+self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] -= self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # re-scale images to be within range [0, 1]
        imgs = (imgs-self.clip_min)/(self.clip_max-self.clip_min)
        imgs = np.clip(imgs, 0, 1)
        # now convert to [-1, 1]
        imgs = (imgs*2)-1
        # convert to tanh-space
        imgs = np.arctanh(imgs*.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        # placeholders for the best l2, score, and image attack found so far
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)+self.clip_min]*batch_size

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestl2 = [1e10]*batch_size
            bestscore = [-1]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l2s, scores, nimg = self.sess.run([self.train,
                                                         self.loss,
                                                         self.l2dist,
                                                         self.output,
                                                         self.newimg])

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and \
                   iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    lab = np.argmax(batchlab[e])
                    if l2 < bestl2[e] and compare(sc, lab):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, lab):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and \
                   bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack

def show(img):
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

import sys
import tensorflow as tf
import numpy as np

MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # abort gradient descent upon first valid solution
LEARNING_RATE = 1e-2    # larger values converge faster to less accurate results
INITIAL_CONST = 1e-3    # the first value of c to start at
LARGEST_CONST = 2e6     # the largest value of c to go up to before giving up
REDUCE_CONST = False    # try to lower c each iteration; faster to set to false
TARGETED = True         # should we target one specific class? or just be wrong?
CONST_FACTOR = 2.0      # f>1, rate at which we increase constant, smaller better

class CarliniWagnerL0:
    def __init__(self, sess, model,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 max_iterations = MAX_ITERATIONS, abort_early = ABORT_EARLY,
                 initial_const = INITIAL_CONST, largest_const = LARGEST_CONST,
                 reduce_const = REDUCE_CONST, const_factor = CONST_FACTOR,
                 independent_channels = False, num_labels=10, shape=(28,28,1)):
        """
        The L_0 optimized attack. 

        Returns adversarial examples for the supplied model.

        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. Should be set to a very small
          value (but positive).
        largest_const: The largest constant to use until we report failure. Should
          be set to a very large value.
        const_factor: The rate at which we should increase the constant, when the
          previous constant failed. Should be greater than one, smaller is better.
        independent_channels: set to false optimizes for number of pixels changed,
          set to true (not recommended) returns number of channels changed.
        """

        self.model = model
        self.sess = sess
        self.shape = tuple([1]+list(shape))
        self.num_labels = num_labels

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor
        self.independent_channels = independent_channels

        self.grad = self.gradient_descent(sess, model)

    def gradient_descent(self, sess, model):
        def compare(x,y):
            if self.TARGETED:
                return x == y
            else:
                return x != y
        shape = self.shape
        
        # the variable to optimize over
        modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

        # the variables we're going to hold, use for efficiency
        canchange = tf.Variable(np.zeros(shape),dtype=np.float32)
        simg = tf.Variable(np.zeros(shape,dtype=np.float32))
        original = tf.Variable(np.zeros(shape,dtype=np.float32))
        timg = tf.Variable(np.zeros(shape,dtype=np.float32))
        tlab = tf.Variable(np.zeros((1,self.num_labels),dtype=np.float32))
        const = tf.placeholder(tf.float32, [])

        # and the assignment to set the variables
        assign_modifier = tf.placeholder(np.float32,shape)
        assign_canchange = tf.placeholder(np.float32,shape)
        assign_simg = tf.placeholder(np.float32,shape)
        assign_original = tf.placeholder(np.float32,shape)
        assign_timg = tf.placeholder(np.float32,shape)
        assign_tlab = tf.placeholder(np.float32,(1,self.num_labels))

        # these are the variables to initialize when we run
        set_modifier = tf.assign(modifier, assign_modifier)
        setup = []
        setup.append(tf.assign(canchange, assign_canchange))
        setup.append(tf.assign(timg, assign_timg))
        setup.append(tf.assign(original, assign_original))
        setup.append(tf.assign(simg, assign_simg))
        setup.append(tf.assign(tlab, assign_tlab))
        
        newimg = (tf.tanh(modifier + simg)/2)*canchange+(1-canchange)*original
        
        output = model(newimg)
        
        real = tf.reduce_sum((tlab)*output,1)
        other = tf.reduce_max((1-tlab)*output - (tlab*10000),1)
        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real+.01)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real-other+.01)

        # sum up the losses
        loss2 = tf.reduce_sum(tf.square(newimg-tf.tanh(timg)/2))
        loss = const*loss1+loss2
            
        outgrad = tf.gradients(loss, [modifier])[0]
        
        # setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train = optimizer.minimize(loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        init = tf.variables_initializer(var_list=[modifier,canchange,simg,
                                                  original,timg,tlab]+new_vars)

        
        def doit(oimgs, labs, starts, valid, CONST):
            # convert to tanh-space
            imgs = np.arctanh(np.array(oimgs)*1.999999)
            starts = np.arctanh(np.array(starts)*1.999999)

            # initialize the variables
            sess.run(init)
            sess.run(setup, {assign_timg: imgs, 
                                    assign_tlab:labs, 
                                    assign_simg: starts, 
                                    assign_original: oimgs,
                                    assign_canchange: valid})

            while CONST < self.LARGEST_CONST:
                # try solving for each value of the constant
                print('try const', CONST)
                for step in range(self.MAX_ITERATIONS):
                    feed_dict={const: CONST}

                    # remember the old value
                    oldmodifier = self.sess.run(modifier)

                    if step%(self.MAX_ITERATIONS//10) == 0:
                        print(step,*sess.run((loss1,loss2),feed_dict=feed_dict))

                    # perform the update step
                    _, works = sess.run([train, loss1], feed_dict=feed_dict)
                        
                    if works < .0001 and (self.ABORT_EARLY or step == CONST-1):
                        # it worked previously, restore the old value and finish
                        self.sess.run(set_modifier, {assign_modifier: oldmodifier})
                        grads, scores, nimg = sess.run((outgrad, output,newimg),
                                                       feed_dict=feed_dict)
                        l2s=np.square(nimg-np.tanh(imgs)/2).sum(axis=(1,2,3))
                        return grads, scores, nimg, CONST

                # we didn't succeed, increase constant and try again
                CONST *= self.const_factor
        return doit
        
    def attack(self, imgs, targets):
        """
        Perform the L_0 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        for i,(img,target) in enumerate(zip(imgs, targets)):
            print("Attack iteration",i)
            r.extend(self.attack_single(img, target))
        return np.array(r)

    def attack_single(self, img, target):
        """
        Run the attack on a single image and label
        """

        # the pixels we can change
        valid = np.ones(self.shape)

        # the previous image
        prev = np.copy(img).reshape(self.shape)
        last_solution = None
        const = self.INITIAL_CONST
    
        while True:
            # try to solve given this valid map
            res = self.grad([np.copy(img)], [target], np.copy(prev), 
                       valid, const)
            if res == None:
                # the attack failed, we return this as our final answer
                print("Final answer",equal_count)
                return last_solution
    
            # the attack succeeded, now we pick new pixels to set to 0
            restarted = False
            gradientnorm, scores, nimg, const = res
            if self.REDUCE_CONST: const /= 2

            #show(nimg[0])
            equal_count = self.shape[1]*self.shape[2]*self.shape[3]-np.sum(np.abs(img-nimg[0])<.0001)
            print("Forced equal:",np.sum(1-valid),
                  "Equal count:",equal_count)
            if np.sum(valid) == 0:
                # if no pixels changed, return 
                return [img]
    
            if self.independent_channels:
                # we are allowed to change each channel independently
                valid = valid.flatten()
                totalchange = abs(nimg[0]-img)*np.abs(gradientnorm[0])
            else:
                # we care only about which pixels change, not channels independently
                # compute total change as sum of change for each channel
                raise
                valid = valid.reshape((self.model.image_size**2,self.model.num_channels))
                totalchange = abs(np.sum(nimg[0]-img,axis=2))*np.sum(np.abs(gradientnorm[0]),axis=2)
            totalchange = totalchange.flatten()

            # set some of the pixels to 0 depending on their total change
            did = 0
            for e in np.argsort(totalchange):
                if np.all(valid[e]):
                    did += 1
                    valid[e] = 0

                    if totalchange[e] > .01:
                        # if this pixel changed a lot, skip
                        break
                    if did >= 9.3*equal_count**.5:
                        # if we changed too many pixels, skip
                        break

            valid = np.reshape(valid,self.shape)
            print("Now forced equal:",np.sum(1-valid))
    
            last_solution = prev = nimg
    
