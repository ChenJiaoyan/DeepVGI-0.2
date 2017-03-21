#! /usr/bin/python

import tensorflow as tf


class Model(object):
    epoch_num = 2000
    class_num = 2
    batch_size = 50

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

    def set_prediction_input(self, imgs):
        self.X_imgs = imgs
        self.X_shape = imgs.shape

    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num

    def set_class_num(self, class_num):
        self.class_num = class_num

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def train_cnn(self):
        ## input
        x_image = tf.placeholder(tf.float32, shape=[None, self.rows, self.cols, self.bands])
        y_ = tf.placeholder(tf.float32, shape=[None, self.class_num])
        tf.add_to_collection("x_image", x_image)
        tf.add_to_collection("y_", y_)

        ## First Layer
        W_conv1 = self.__weight_variable([4, 4, self.bands, 16])
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
        tf.add_to_collection("keep_prob", keep_prob)

        # Readout Layer
        W_fc2 = self.__weight_variable([512, 2])
        b_fc2 = self.__bias_variable([2])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        tf.add_to_collection("y_conv", y_conv)

        init_op = tf.global_variables_initializer()
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.add_to_collection("accuracy", accuracy)

        ## Debug Ops
        y_conv_shape = tf.shape(y_conv)
        y_shape = tf.shape(y_)
        h_fc1_drop_shape = tf.shape(h_fc1_drop)
        h_fc1_shape = tf.shape(h_fc1)
        h_pool2_shape = tf.shape(h_pool2)
        h_conv2_shape = tf.shape(h_conv2)

        saver = tf.train.Saver()
        saver.export_meta_graph('../data/model/%s.meta' % self.name)

        print '#################  start learning  ####################'
        with tf.Session() as sess:
            sess.run(init_op)
            for i in range(self.epoch_num):
                ran = self.__get_batch(self.sample_size, i, self.batch_size)
                shape1 = sess.run(y_conv_shape,
                                  feed_dict={x_image: self.X_imgs[ran], y_: self.Y_labels[ran], keep_prob: 0.5})
                shape2 = sess.run(y_shape,
                                  feed_dict={x_image: self.X_imgs[ran], y_: self.Y_labels[ran], keep_prob: 0.5})
                shape3 = sess.run(h_fc1_drop_shape,
                                  feed_dict={x_image: self.X_imgs[ran], y_: self.Y_labels[ran], keep_prob: 0.5})
                shape4 = sess.run(h_fc1_shape,
                                  feed_dict={x_image: self.X_imgs[ran], y_: self.Y_labels[ran], keep_prob: 0.5})
                shape5 = sess.run(h_pool2_shape,
                                  feed_dict={x_image: self.X_imgs[ran], y_: self.Y_labels[ran], keep_prob: 0.5})
                shape6 = sess.run(h_conv2_shape,
                                  feed_dict={x_image: self.X_imgs[ran], y_: self.Y_labels[ran], keep_prob: 0.5})
                print 'y_conv_shape: '
                print shape1
                print 'y_shape: '
                print shape2
                print 'h_fc1_drop_shape: '
                print shape3
                print 'h_fc1_shape: '
                print shape4
                print 'h_pool2_shape'
                print shape5
                print 'h_conv2_shape'
                print shape6
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(session=sess,
                                                   feed_dict={x_image: self.X_imgs[ran], y_: self.Y_labels[ran],
                                                              keep_prob: 1.0})
                    print("step %d, training accuracy %g \n" % (i, train_accuracy))
                train_step.run(session=sess,
                               feed_dict={x_image: self.X_imgs[ran, :], y_: self.Y_labels[ran], keep_prob: 0.5})
            saver.save(sess, '../data/model/%s.ckpt' % self.name)
        print '#################  end learning  ####################'
        print 'model saved in %s.meta and %s.ckpt' % (self.name, self.name)

    def predict(self):
        saver = tf.train.import_meta_graph('../data/%s.meta' % self.name)
        y_conv = tf.get_collection("y_conv")[0]
        x_image = tf.get_collection("x_image")[0]
        keep_prob = tf.get_collection("keep_prob")[0]
        prediction = tf.argmax(y_conv, 1)
        prob = tf.divide(y_conv[:, 1], tf.add(y_conv[:, 0] + y_conv[:, 1]))
        print '#################  start predicting  ####################'
        with tf.Session() as sess:
            saver.restore(sess, "../data/model/%s.ckpt" % self.name)
            label_results = sess.run(prediction, feed_dict={x_image: self.X_imgs, keep_prob: 1.0})
            print label_results
            prob_results = sess.run(prob, feed_dict={x_image: self.X_imgs, keep_prob: 1.0})
            print prob_results
        print '#################  end predicting  ####################'

    def evaluate(self):
        saver = tf.train.import_meta_graph('../data/%s.meta' % self.name)
        accuracy = tf.get_collection("accuracy")[0]
        x_image = tf.get_collection("x_image")[0]
        y_ = tf.get_collection("y_")[0]
        keep_prob = tf.get_collection("keep_prob")[0]
        print '#################  start evaluation  ####################'
        with tf.Session() as sess:
            saver.restore(sess, "../data/model/%s.ckpt" % self.name)
            acc = accuracy.eval(session=sess, feed_dict={x_image: self.X_imgs, y_: self.Y_labels,
                                                         keep_prob: 1.0})
            print("testing accuracy %g \n" % acc)
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
