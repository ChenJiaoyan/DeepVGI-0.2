#! /usr/bin/python

# (1) train CNN with MapSwipe labeled images; (2) evaluate the CNN with testing MapSwipe images
import os
if not os.getcwd().endswith('exp1'):
    os.chdir('exp1')

import sys
import random
import gc
import numpy as np
from scipy import misc

sys.path.append("../lib")
import NN_Model
import MapSwipe
import Parameters
import FileIO

sample_dir = '../samples0/'


## Positive: MapSwipe,  Negative: MapSwipe
def read_train_sample(n1, n0):
    client = MapSwipe.MSClient()
    MS_train_p = client.MS_train_positive()
    MS_train_n = client.MS_train_negative()

    print 'MS_train_p: %d \n' % len(MS_train_p)
    print 'MS_train_n: %d \n' % len(MS_train_n)

    if len(MS_train_p) < n1:
        print 'n1 is set too large'
        sys.exit()

    if len(MS_train_n) < n0:
        print 'n0 is set too large'
        sys.exit()

    img_X1, img_X0 = np.zeros((n1, 256, 256, 3)), np.zeros((n0, 256, 256, 3))

    MS_train_p = random.sample(MS_train_p, n1)
    for i, img in enumerate(MS_train_p):
        img_X1[i] = misc.imread(os.path.join(sample_dir, 'train/MS_record/', img))

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


if __name__ == '__main__':
    evaluate_only, external_test, tr_n1, tr_n0, tr_b, tr_e, tr_t, te_n, nn = Parameters.deal_args(sys.argv[1:])

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
    img_X2, Y2 = FileIO.read_valid_sample(te_n)
    m.set_evaluation_input(img_X2, Y2)
    m.evaluate()
    del img_X2, Y2
    gc.collect()

    if external_test:
        print '--------------- Evaluation on External Test Samples ---------------'
        img_X3, Y3 = FileIO.read_external_test_sample()
        m.set_evaluation_input(img_X3, Y3)
        m.evaluate()
        del img_X3, Y3
        gc.collect()
