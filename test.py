#! /usr/bin/python
import sys
sys.path.append("lib")
import MapSwipe

import tensorflow as tf


def store_test():
    w1 = tf.Variable(tf.truncated_normal(shape=[10]), name='w1')
    w2 = tf.Variable(tf.truncated_normal(shape=[10]), name='w2')
    w12 = tf.add(w1, w2)
    tf.add_to_collection("ws", w1)
    tf.add_to_collection("ws", w2)
    tf.add_to_collection("ws", w12)

    saver = tf.train.Saver()

    saver.export_meta_graph('/tmp/CNN_New.meta')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(w12)
        saver.save(sess, '/tmp/CNN_New.ckpt')
    print 'saved'


def restore_test():
    saver = tf.train.import_meta_graph('/tmp/CNN_New.meta')
    w1 = tf.get_collection("ws")[0]
    w2 = tf.get_collection("ws")[1]
    w12 = tf.get_collection("ws")[2]
    w121 = tf.add(w12, w1)
    with tf.Session() as sess:
        saver.restore(sess, "/tmp/CNN_New.ckpt")
        print("Model restored.")
        print w1.eval()
        print ''
        print w2.eval()
        print ''
        print w12.eval()
        print ''
        r = sess.run(w121)
        print r

#store_test()
#restore_test()

client = MapSwipe.MSClient()
s_n_imgs = client.read_n_images()
print s_n_imgs
print 'negative images: %d' % len(s_n_imgs)
