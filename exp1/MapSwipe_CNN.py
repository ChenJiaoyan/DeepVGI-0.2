#! /usr/bin/python

import os
import sys
import numpy as np
import csv
import tensorflow as tf

from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class Model(object):
    IMG_DIR = '../data/exp1'
    BAND_N = 3
    TAG = 'mapswipe'  # 'mapswipe' or 'osm'
    IMAGE_L = 256

    def __init__(self):
        self.X = None
        self.y = None

    def __positive_image_mapswipe(self):
        p_image, b_image, m_image = [], [], []
        msfile = '../data/project_922.csv'
        # open csv and read columns
        with open(msfile) as csvfile:
            reader = csv.DictReader(csvfile)
            for r in reader:
                image = str(r['task_x']) + '_' + str(r['task_y']) + '_18.jpeg'
                if r['bad_imagery_count'] == 0 and r['yes_count'] > 0:
                    p_image.append(image)
                if r['bad_imagery_count'] > 0:
                    b_image.append(image)
                if r['maybe_count'] > 0:
                    m_image.append(image)
        return p_image, b_image, m_image

    def sample(self):  # class labels for images and return X, y
        if Model.TAG == 'mapswipe':
            p_image, b_image, m_image = self.__positive_image_mapswipe()
        elif Model.TAG == 'osm':
            print 'currently can not work of osm tag'
            sys.exit()
        else:
            print 'please enter right tag source'
            sys.exit()
        images = os.listdir(Model.IMG_DIR)
        L = Model.IMAGE_L * Model.IMAGE_L * Model.BAND_N
        tmp_X, tmp_y = np.zeros((len(images), L)), np.zeros((len(images)))
        i = 0
        for image in images:
            if image in b_image and image in m_image:
                print 'Ignore: ' + image
            elif image in p_image:
                m = io.imread(os.path.join(Model.IMG_DIR, image))  # load an image from filename
                # m is an image array, The different colour bands/channels are stored in the third dimension,
                # an RGB-image MxNx3
                tmp_X[i], tmp_y[i] = m[:, :, 0:Model.BAND_N].reshape((1, L)), 1  # make positive image label 1
                i = i + 1
                print 'Positive: ' + image
            else:
                m = io.imread(os.path.join(Model.IMG_DIR, image))
                tmp_X[i], tmp_y[i] = m[:, :, 0:Model.BAND_N].reshape((1, L)), 0  # make negative image label 0
                i = i + 1
                print 'Negative: ' + image
        self.X, self.y = tmp_X[0:i], tmp_y[0:i]
        print 'self.X.shape: (%d, %d)' % self.X.shape
        print 'self.y.shape: (%d)' % self.y.shape
        print 'Positive sample #: %d, Negative sample #: %d' % \
              (np.where(self.y == 1)[0].shape[0], np.where(self.y == 0)[0].shape[0])

    def __get_batch(self, l, i, n):
        if l % n == 0:
            m = l / n
            buttom, top = i % m * n, i % m * n + n
        else:
            m = l / n + 1
            buttom = i % m * n
            if buttom + n > l:
                top = l
            else:
                top = buttom + n
        return range(buttom, top)

    def __weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)  # Outputs random values from a truncated normal distribution.
        # shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
        return tf.Variable(initial)

    def __bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def __conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # input(x) is a tensor, filter(W) must have the same type as input

    def __max_pool_4x4(self, x):
        return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], \
                              padding='SAME')

    # value(x) a 4D tensor with shape [batch, height, width, channels] and type tf.float32
    # ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.

    def CNN_CV(self):
        # Input
        x = tf.placeholder(tf.float32, shape=[None, \
                                              Model.IMAGE_L * Model.IMAGE_L * Model.BAND_N])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])
        x_image = tf.reshape(x, [-1, Model.IMAGE_L, Model.IMAGE_L,
                                 Model.BAND_N])  # -1 means "figure this part out for me".

        # First Layer
        W_conv1 = self.__weight_variable([12, 12, Model.BAND_N, 32])
        b_conv1 = self.__bias_variable([32])
        h_conv1 = tf.nn.relu(self.__conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.__max_pool_4x4(h_conv1)

        # Second Layer
        W_conv2 = self.__weight_variable([12, 12, 32, 64])
        b_conv2 = self.__bias_variable([64])
        h_conv2 = tf.nn.relu(self.__conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.__max_pool_4x4(h_conv2)

        # Third Layer
        W_fc1 = self.__weight_variable([32 * 32 * 64, 1024])
        b_fc1 = self.__bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Readout Layer
        W_fc2 = self.__weight_variable([1024, 2])
        b_fc2 = self.__bias_variable([2])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # operations
        cross_entropy = tf.reduce_mean( \
            tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))  # Loss function
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        prediction = tf.argmax(y_conv, 1)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Train and Evaluate
        # X_train, X_test, y_train, y_test = train_test_split(\
        #      self.X,self.y,test_size=0.2, random_state=0)
        # Y_train,Y_test = np.eye(2)[y_train.astype(int)],np.eye(2)[y_test.astype(int)]

        # K folde CV
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(self.X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
        Y_train, Y_test = np.eye(2)[y_train.astype(int)], np.eye(2)[y_test.astype(int)]

        sess = tf.Session()
        print '     training ...'
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            ran = self.__get_batch(X_train.shape[0], i, 30)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={ \
                    x: X_train[ran], y_: Y_train[ran], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_step.run(session=sess, feed_dict={x: X_train[ran, :], \
                                                    y_: Y_train[ran], keep_prob: 0.5})

        print '     evaluating ...'
        n = X_test.shape[0]
        batch = 25
        m = int(np.ceil(n / float(batch)))
        results = np.zeros((n), dtype=int)
        for i in range(0, m - 1):
            print "testing batch #: %d" % i
            b, t = i * batch, (i + 1) * batch
            results_i = sess.run(prediction, feed_dict={x: X_test[b:t], keep_prob: 1.0})
            results[b:t] = results_i
        print 'testing batch #: %d' % (m - 1)
        b, t = (m - 1) * batch, n
        results_i = sess.run(prediction, feed_dict={x: X_test[b:t], keep_prob: 1.0})
        results[b:t] = results_i

        print 'testing accuracy: %.4f' % accuracy_score(y_test, results)

        # Close Session
        sess.close()


m = Model()
print 'sampling...'
m.sample()
print 'cross validation...'
m.CNN_CV()
