#! /usr/bin/python

# (1) train CNN with OSM labeled images; (2) evaluate the CNN with testing MapSwipe images
import os
if not os.getcwd().endswith('exp2'):
    os.chdir('exp2')

import sys
import random
import gc
import numpy as np
from scipy import misc

sys.path.append("../lib")
import NN_Model
import FileIO
import MapSwipe
import Parameters

sample_dir = '../samples0/'


def read_train_sample(n1, n0):
    client = MapSwipe.MSClient()
    MS_train_n = client.MS_train_negative()
    MS_train_r = client.MS_train_record()
    MS_train = MS_train_r + MS_train_n

    task_w = FileIO.osm_building_weight();
    OSM_train_p = list(set(task_w.keys()).intersection(set(MS_train)))
    MS_train_n = list(set(MS_train_n).difference(set(OSM_train_p)))

    print 'OSM_train_p: %d \n' % len(OSM_train_p)
    print 'MS_train_n: %d \n' % len(MS_train_n)

    if len(OSM_train_p) < n1:
        print 'n1 is set too large'
        sys.exit()

    if len(MS_train_n) < n0:
        print 'n0 is set too large'
        sys.exit()

    img_X1, img_X0 = np.zeros((n1, 256, 256, 3)), np.zeros((n0, 256, 256, 3))
    OSM_train_p = random.sample(OSM_train_p, n1)
    for i, img in enumerate(OSM_train_p):
        if img in MS_train_r:
            img_X1[i] = misc.imread(os.path.join(sample_dir, 'train/MS_record/', img))
        else:
            img_X1[i] = misc.imread(os.path.join(sample_dir, 'train/MS_negative/', img))

    MS_train_n = random.sample(MS_train_n, n0)
    for i, img in enumerate(MS_train_n):
        img_X0[i] = misc.imread(os.path.join(sample_dir, 'train/MS_negative/', img))

    label = np.zeros((n1 + n0, 2))
    label[0:n1, 1] = 1
    label[n1:(n1 + n0), 0] = 1

    j = range(n1 + n0)
    random.shuffle(j)
    X = np.concatenate((img_X1, img_X0))
    return X[j], label[j]


if __name__ == '__main__':
    evaluate_only, external_test, tr_n1, tr_n0, tr_b, tr_e, tr_t, te_n, nn = Parameters.deal_args(sys.argv[1:])

    print '--------------- Read Samples ---------------'
    img_X, Y = read_train_sample(tr_n1, tr_n0)

    m = NN_Model.Model(img_X, Y, nn + '_DL_OSMMS')
    if not evaluate_only:
        print '--------------- Training on OSM Labels---------------'
        m.set_batch_size(tr_b)
        m.set_epoch_num(tr_e)
        m.set_thread_num(tr_t)
        m.train(nn)
        print '--------------- Evaluation on Training Samples ---------------'
        m.evaluate()
    del img_X, Y
    gc.collect()

    print '--------------- Evaluation on Validation Samples ---------------'
    img_X2, Y2 = FileIO.read_urban_osm_valid(te_n)
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
