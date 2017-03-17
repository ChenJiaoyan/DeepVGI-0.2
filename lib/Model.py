#! /usr/bin/python

import tensorflow as tf


class Model(object):
    epoch_num = 2000

    def __init__(self, imgs, labels, class_num=2):
        self.X_imgs = imgs
        self.Y_labels = labels
        self.X_shape = imgs.shape
        self.class_num = class_num

    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num

    def train_cnn(self):
        ## input
        x_image = tf.placeholder(tf.float32, shape=[None, self.X_shape[1], self.X_shape[2], self.X_shape[3]])
        y_ = tf.placeholder(tf.float32, shape=[None, self.class_num])

        ## First Layer
        W_conv1 = self.__weight_variable([4, 4, self.img_shape[3], 16])
        b_conv1 = self.__bias_variable([16])
        h_conv1 = tf.nn.relu(self.__conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.__max_pool_2x2(h_conv1)

        ## Second Layer
        W_conv2 = self.__weight_variable([4, 4, 16, 32])
        b_conv2 = self.__bias_variable([32])
        h_conv2 = tf.nn.relu(self.__conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.__max_pool_2x2(h_conv2)

        ## Third Layer
        W_fc1 = self.__weight_variable([16 * 16 * 32, 512])
        b_fc1 = self.__bias_variable([512])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        ## Dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Readout Layer
        W_fc2 = self.__weight_variable([512, 2])
        b_fc2 = self.__bias_variable([2])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        ## Operation
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        prediction = tf.argmax(y_conv, 1)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.Session()
        print '     training ...'
        sess.run(tf.initialize_all_variables())
        for i in range(self.epoch_num):
            ran = self.__get_batch(self.X_shape[0], i, 50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(session=sess,
                                               feed_dict={x_image: self.X_imgs[ran], y_: self.Y_labels[ran],
                                                          keep_prob: 1.0})
                print("step %d, training accuracy %g \n" % (i, train_accuracy))
            train_step.run(session=sess, feed_dict={x_image: self.X_imgs[ran, :], y_: self.Y_labels[ran], keep_prob: 0.5})

    def __get_batch(self, l, i, n):
        if l % n == 0:
            m = l / n
            bottom, top = i % m * n, i % m * n + n
        else:
            m = l / n + 1
            bottom = i % m * n
            if bottom + n > l:
                top = l
            else:
                top = bottom + n
        return range(bottom, top)

    def __weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def __bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def __conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def __max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
