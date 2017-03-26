#! /usr/bin/python

import tensorflow as tf
import numpy as np
from sklearn import metrics
import math


class Model(object):
    epoch_num = 2000
    class_num = 2
    batch_size = 50
    thread_num = 4

    def __init__(self, imgs, labels, name='CNN_Default'):
        # X_imgs shape: img_number * Row * Col * RGB_bands
        self.X_imgs = imgs
        self.X_shape = imgs.shape
        self.Y_labels = labels
        self.name = name
        self.sample_size = imgs.shape[0]
        self.rows = imgs.shape[1]
        self.cols = imgs.shape[2]
        self.bands = imgs.shape[3]

    def set_evaluation_input(self, imgs, labels):
        self.X_imgs = imgs
        self.Y_labels = labels
        self.X_shape = imgs.shape
        self.sample_size = imgs.shape[0]
        self.rows = imgs.shape[1]
        self.cols = imgs.shape[2]
        self.bands = imgs.shape[3]

    def set_prediction_input(self, imgs):
        self.X_imgs = imgs
        self.X_shape = imgs.shape
        self.sample_size = imgs.shape[0]
        self.rows = imgs.shape[1]
        self.cols = imgs.shape[2]
        self.bands = imgs.shape[3]

    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num

    def set_class_num(self, class_num):
        self.class_num = class_num

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_thread_num(self, thread_num):
        self.thread_num = thread_num

    def train(self, nn):
        if nn == 'segnet':
            self.train_segnet()
        elif nn == 'lenet':
            self.train_lenet()
        elif nn == 'alexnet':
            self.train_alexnet()

    def train_segnet(self):
        a = ''

    def train_alexnet(self):
        ## input
        x_image = tf.placeholder(tf.float32, shape=[None, self.rows, self.cols, self.bands])
        y_ = tf.placeholder(tf.float32, shape=[None, self.class_num])
        tf.add_to_collection("x_image", x_image)
        tf.add_to_collection("y_", y_)

        ## Conv1, Pool1, Norm1
        W_conv1 = self.__weight_variable([11, 11, self.bands, 96])
        b_conv1 = self.__bias_variable([96])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 4, 4, 1], padding='SAME') + b_conv1)
        h_pool1 = self.__max_pool_4x4(h_conv1)
        h_norm1 = self.__norm_4(h_pool1)

        ## Conv2, Pool2, Norm2
        W_conv2 = self.__weight_variable([5, 5, 96, 256])
        b_conv2 = self.__bias_variable([256])
        h_conv2 = tf.nn.relu(self.__conv2d(h_norm1, W_conv2) + b_conv2)
        h_pool2 = self.__max_pool_4x4(h_conv2)
        h_norm2 = self.__norm_4(h_pool2)

        ## Conv3
        W_conv3 = self.__weight_variable([3, 3, 256, 384])
        b_conv3 = self.__bias_variable([384])
        h_conv3 = tf.nn.relu(self.__conv2d(h_norm2, W_conv3) + b_conv3)

        ## Conv4
        W_conv4 = self.__weight_variable([3, 3, 384, 384])
        b_conv4 = self.__bias_variable([384])
        h_conv4 = tf.nn.relu(self.__conv2d(h_conv3, W_conv4) + b_conv4)

        ## Conv5, 5
        W_conv5 = self.__weight_variable([3, 3, 384, 256])
        b_conv5 = self.__bias_variable([256])
        h_conv5 = tf.nn.relu(self.__conv2d(h_conv4, W_conv5) + b_conv5)
        h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        ## FC6
        W_fc6 = self.__weight_variable([2 * 2 * 256, 4096])
        b_fc6 = self.__bias_variable([4096])
        h_pool5_flat = tf.reshape(h_pool5, [-1, 2 * 2 * 256])
        h_fc6 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc6) + b_fc6)

        ## FC7, Dropout
        W_fc7 = self.__weight_variable([4096, 4096])
        b_fc7 = self.__bias_variable([4096])
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        keep_prob = tf.placeholder(tf.float32)
        h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)
        tf.add_to_collection("keep_prob", keep_prob)

        ## FC8 (Readout Layer)
        W_fc8 = self.__weight_variable([4096, 2])
        b_fc8 = self.__bias_variable([2])
        y_conv = tf.nn.relu(tf.matmul(h_fc7_drop, W_fc8) + b_fc8)
        tf.add_to_collection("y_conv", y_conv)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.add_to_collection("accuracy", accuracy)

        saver = tf.train.Saver()
        saver.export_meta_graph('../data/model/%s.meta' % self.name)
        self.learn(saver, x_image, y_, keep_prob, train_step, accuracy)

    def train_lenet(self):
        ## input
        x_image = tf.placeholder(tf.float32, shape=[None, self.rows, self.cols, self.bands])
        y_ = tf.placeholder(tf.float32, shape=[None, self.class_num])
        tf.add_to_collection("x_image", x_image)
        tf.add_to_collection("y_", y_)

        ## First Layer
        W_conv1 = self.__weight_variable([8, 8, self.bands, 16])
        b_conv1 = self.__bias_variable([16])
        h_conv1 = tf.nn.relu(self.__conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.__max_pool_4x4(h_conv1)

        ## Second Layer
        W_conv2 = self.__weight_variable([4, 4, 16, 32])
        b_conv2 = self.__bias_variable([32])
        h_conv2 = tf.nn.relu(self.__conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.__max_pool_4x4(h_conv2)

        ## Third Layer
        W_fc1 = self.__weight_variable([16 * 16 * 32, 512])
        b_fc1 = self.__bias_variable([512])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        ## Dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        tf.add_to_collection("keep_prob", keep_prob)

        # Readout Layer
        W_fc2 = self.__weight_variable([512, 2])
        b_fc2 = self.__bias_variable([2])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        tf.add_to_collection("y_conv", y_conv)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.add_to_collection("accuracy", accuracy)

        saver = tf.train.Saver()
        saver.export_meta_graph('../data/model/%s.meta' % self.name)
        self.learn(saver, x_image, y_, keep_prob, train_step, accuracy)

    def learn(self, saver, x_image, y_, keep_prob, train_step_op, accuracy_op):
        print '#################  start learning  ####################'
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.thread_num)) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i in range(self.epoch_num):
                ran = self.__get_batch(self.sample_size, i, self.batch_size)
                train_step_op.run(session=sess,
                                  feed_dict={x_image: self.X_imgs[ran, :], y_: self.Y_labels[ran], keep_prob: 0.5})
                if i % 100 == 0:
                    train_accuracy = accuracy_op.eval(session=sess,
                                                      feed_dict={x_image: self.X_imgs[ran], y_: self.Y_labels[ran],
                                                                 keep_prob: 1.0})
                    print("epoch %d, training accuracy %f \n" % (i, train_accuracy))
            saver.save(sess, '../data/model/%s.ckpt' % self.name)
        print '#################  end learning  ####################'

    def evaluate(self):
        saver = tf.train.import_meta_graph('../data/model/%s.meta' % self.name)
        x_image = tf.get_collection("x_image")[0]
        y_ = tf.get_collection("y_")[0]
        keep_prob = tf.get_collection("keep_prob")[0]
        y_conv = tf.get_collection("y_conv")[0]
        print '#################  start evaluation  ####################'
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.thread_num)) as sess:
            saver.restore(sess, "../data/model/%s.ckpt" % self.name)
            batch = 200
            label_pred, score = np.zeros(self.sample_size), np.zeros(self.sample_size)
            for i in range(int(math.ceil(self.sample_size / float(batch)))):
                i1 = i * batch
                i2 = (i + 1) * batch if (i + 1) * batch < self.sample_size else self.sample_size
                print 'predict...  %d to %d \n' % (i1, i2)
                label_pred[i1:i2], score[i1:i2] = sess.run([tf.argmax(y_conv, 1), tf.nn.softmax(y_conv)[:, 1]],
                                                           feed_dict={x_image: self.X_imgs[i1:i2],
                                                                      y_: self.Y_labels[i1:i2], keep_prob: 1.0})
            label_true = np.argmax(self.Y_labels, 1)
            print 'Accuracy: %f \n' % metrics.accuracy_score(label_true, label_pred)
            print 'Precision: %f \n' % metrics.precision_score(label_true, label_pred)
            print 'Recall: %f \n' % metrics.recall_score(label_true, label_pred)
            print 'F1 Score: %f \n' % metrics.f1_score(label_true, label_pred)
            print 'ROC_AUC: %f \n' % metrics.roc_auc_score(label_true, score)
        print '#################  end evaluation  ####################'

    @staticmethod
    def __get_batch(l, i, n):
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

    @staticmethod
    def __weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def __bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def __conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def __max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def __max_pool_4x4(x):
        return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    @staticmethod
    def __norm_4(x):
        return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
