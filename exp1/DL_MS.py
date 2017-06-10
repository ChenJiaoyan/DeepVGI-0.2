#! /usr/bin/python

# (1) train CNN with MapSwipe labeled images; (2) evaluate the CNN with testing MapSwipe images
import os
import sys
import random
import getopt
import gc
import numpy as np
from scipy import misc

sys.path.append("../lib")
import NN_Model
import MapSwipe

sample_dir = '../samples0/'


def read_test_sample(n):
    client = MapSwipe.MSClient()
    MS_valid_p = client.MS_valid_positive()
    MS_valid_n = client.MS_valid_negative()

    print 'MS_valid_p: %d \n' % len(MS_valid_p)
    print 'MS_valid_n: %d \n' % len(MS_valid_n)

    if len(MS_valid_p) <= n/2 or len(MS_valid_p) <= n/2:
        print 'n is set too large; use all the samples for testing'
        n = len(MS_valid_p) * 2

    img_X1, img_X0 = np.zeros((n/2, 256, 256, 3)), np.zeros((n/2, 256, 256, 3))
    MS_valid_p = random.sample(MS_valid_p, n/2)
    for i, img in enumerate(MS_valid_p):
        img_X1[i] = misc.imread(os.path.join(sample_dir, 'valid/MS_positive/', img))

    MS_valid_n = random.sample(MS_valid_n, n/2)
    for i, img in enumerate(MS_valid_n):
        img_X1[i] = misc.imread(os.path.join(sample_dir, 'valid/MS_negative/', img))

    X = np.concatenate((img_X1[0:n/2], img_X0[0:n/2]))

    label = np.zeros((n, 2))
    label[0:n/2, 1] = 1
    label[n/2:n, 0] = 1

    return X, label


## Positive: MapSwipe,  Negative: MapSwipe
def read_train_sample(n1, n0):
    client = MapSwipe.MSClient()
    MS_train_p = client.MS_train_positive()
    MS_train_n = client.MS_train_negative()

    print 'MS_train_p: %d \n' % len(MS_train_p)
    print 'MS_train_n: %d \n' % len(MS_train_n)

    if len(MS_train_p) <= n1:
        print 'n1 is set too large'
        sys.exit()

    if len(MS_train_n) <= n0:
        print 'n0 is set too large'
        sys.exit()

    img_X1, img_X0 = np.zeros((n1, 256, 256, 3)), np.zeros((n0, 256, 256, 3))

    MS_train_p = random.sample(MS_train_p, n1)
    for i, img in enumerate(MS_train_p):
        img_X1[i] = misc.imread(os.path.join(sample_dir, 'train/MS_positive/', img))

    MS_train_n = random.sample(MS_train_n, n0)
    for i, img in enumerate(MS_train_n):
        img_X0[i] = misc.imread(os.path.join(sample_dir, 'train/MS_negative/', img))

    X = np.concatenate((img_X1[0:n1], img_X0[0:n0]))

    label = np.zeros((n1 + n0, 2))
    label[0:n1, 1] = 1
    label[n1:(n1 + n0), 0] = 1

    j = range(n1 + n0)
    random.shuffle(j)
    return X[j], label[j]


def deal_args(my_argv):
    v, n1, n0, b, e, t, c, z = False, 100, 100, 30, 1000, 3, 0, 200
    m = 'lenet'
    try:
        opts, args = getopt.getopt(my_argv, "vhy:n:b:e:t:c:z:m:",
                                   ["p_sample_size=", "n_sample_size=", "batch_size=", "epoch_num=", "thread_num=",
                                    "cv_round=", 'test_size=', 'network_model='])
    except getopt.GetoptError:
        print 'DL_MS.py -v -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, ' \
              '-c <cv_round>, -z <test_size>, -m <network_model>'
        print 'default settings: v=%s, n1=%d, n0=%d, b=%d, e=%d, t=%d, c=%d, z=%d, m=%s' % (v, n1, n0, b, e, t, c, z, m)
    for opt, arg in opts:
        if opt == '-h':
            print 'DL_MS.py -v -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, ' \
                  '-c <cv_round>, -z <test_size>, -m <network_model>'
            sys.exit()
        elif opt == '-v':
            v = True
        elif opt in ("-y", "--p_sample_size"):
            n1 = int(arg)
        elif opt in ("-n", "--n_sample_size"):
            n0 = int(arg)
        elif opt in ("-b", "--batch_size"):
            b = int(arg)
        elif opt in ("-e", "--epoch_num"):
            e = int(arg)
        elif opt in ("-t", "--thread_num"):
            t = int(arg)
        elif opt in ("-c", "--cv_round"):
            c = int(arg)
        elif opt in ("-z", "--test_size"):
            z = int(arg)
        elif opt in ("-m", "--network_model"):
            m = arg
    print 'settings: v=%s, n1=%d, n0=%d, b=%d, e=%d, t=%d, c=%d, z=%d, m=%s' % (v, n1, n0, b, e, t, c, z, m)
    return v, n1, n0, b, e, t, c, z, m


if __name__ == '__main__':
    evaluate_only, tr_n1, tr_n0, tr_b, tr_e, tr_t, cv_i, te_n, nn = deal_args(sys.argv[1:])

    print '--------------- Read Samples ---------------'
    img_X, Y = read_train_sample(tr_n1, tr_n0)

    m = NN_Model.Model(img_X, Y, nn + '_ZY')
    if not evaluate_only:
        print '--------------- Training ---------------'
        m.set_batch_size(tr_b)
        m.set_epoch_num(tr_e)
        m.set_thread_num(tr_t)
        m.train(nn)
        print '--------------- Evaluation on Training Samples ---------------'
        m.evaluate()
    del img_X, Y
    gc.collect()

    print '--------------- Evaluation on Validation Samples ---------------'
    img_X2, Y2 = read_test_sample(te_n)
    m.set_evaluation_input(img_X2, Y2)
    m.evaluate()
