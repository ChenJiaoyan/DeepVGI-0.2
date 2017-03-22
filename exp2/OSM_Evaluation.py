#! /usr/bin/python

# (1) train CNN with OSM labeled images; (2) evaluate the CNN with testing MapSwipe images
import os
import sys
import random
import getopt
import numpy as np
from scipy import misc

sys.path.append("../lib")
import NN_Model
import FileIO
import MapSwipe


def osm_building_weight():
    task_w = {}
    osm_buildings = FileIO.csv_reader("../data/buildings.csv")
    for row in osm_buildings:
        task_x = row['task_x']
        task_y = row['task_y']
        k = '%s-%s' % (task_x, task_y)
        task_w[k] = 1
    return task_w


def read_test_sample(n, test_imgs, ms_p_imgs, ms_n_imgs):
    img_X = np.zeros((n, 256, 256, 3))
    label = np.zeros((n, 2))
    img_dir = '../data/image_project_922/'
    random.shuffle(test_imgs)
    i = 0
    for img in test_imgs:
        if i >= n:
            break
        img_X[i] = misc.imread(os.path.join(img_dir, img))
        i1, i2 = img.index('-'), img.index('.')
        task_x, task_y = img[0:i1], img[(i1 + 1):i2]
        if [int(task_x), int(task_y)] in ms_p_imgs:
            label[i, 1] = 1
            i += 1
        elif [int(task_x), int(task_y)] in ms_n_imgs:
            label[i, 0] = 1
            i += 1
    return img_X, label


def read_train_sample(n1, n0, train_imgs):
    img_X1, img_X0 = np.zeros((n1, 256, 256, 3)), np.zeros((n0, 256, 256, 3))
    label = np.zeros((n1 + n0, 2))

    task_w = osm_building_weight();
    img_dir = '../data/image_project_922/'
    osm_imgs, none_osm_imgs = [], []
    for img in train_imgs:
        i1, i2 = img.index('-'), img.index('.')
        task_x, task_y = img[0:i1], img[(i1 + 1):i2]
        k = '%s-%s' % (task_x, task_y)
        if task_w.has_key(k):
            osm_imgs.append(img)
        else:
            none_osm_imgs.append(img)

    osm_imgs = random.sample(osm_imgs, n1)
    for i, img in enumerate(osm_imgs):
        img_X1[i] = misc.imread(os.path.join(img_dir, img))
    label[0:n1, 1] = 1

    none_osm_imgs = random.sample(none_osm_imgs, n0)
    for i, img in enumerate(none_osm_imgs):
        img_X0[i] = misc.imread(os.path.join(img_dir, img))
    label[n1:(n1 + n0), 0] = 1

    j = range(n1 + n0)
    random.shuffle(j)
    X = np.concatenate((img_X1, img_X0))
    return X[j], label[j]


def deal_args(my_argv):
    n1, n0, b, e, t, c = 200, 200, 30, 1000, 4, 0
    try:
        opts, args = getopt.getopt(my_argv, "hy:n:b:e:t:c:",
                                   ["p_sample_size=", "n_sample_size=", "batch_size=", "epoch_num=", "thread_num=",
                                    "cv_round="])
    except getopt.GetoptError:
        print 'OSM_Evaluation.py -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, -c <cv_round>'
        print 'use the default settings: n1=%d, n0=%d, b=%d, e=%d, t=%d, c=%d' % (n1, n0, b, e, t, c)
    for opt, arg in opts:
        if opt == '-h':
            print 'OSM_Evaluation.py -n1 <p_sample_size> -n0 <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, -c <cv_round>'
            sys.exit()
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
    print 'settings: n1=%d, n0=%d, b=%d, e=%d, t=%d, c=%d' % (n1, n0, b, e, t, c)
    return n1, n0, b, e, t, c


if __name__ == '__main__':
    tr_n1, tr_n0, tr_b, tr_e, tr_t, cv_i = deal_args(sys.argv[1:])
    te_n = 1000
    cv_n = 4

    print '--------------- Read Samples ---------------'
    client = MapSwipe.MSClient()
    train_imgs, test_imgs = client.imgs_cross_validation(cv_i, cv_n)
    img_X, Y = read_train_sample(tr_n1, tr_n0, train_imgs)
    ms_p_imgs = client.read_p_images()
    ms_n_imgs = client.read_n_images()
    img_X2, Y2 = read_test_sample(te_n, test_imgs, ms_p_imgs, ms_n_imgs)

    print '--------------- Training on OSM Labels---------------'
    m = NN_Model.Model(img_X, Y, 'CNN_JY')
    m.set_batch_size(tr_b)
    m.set_epoch_num(tr_e)
    m.set_thread_num(tr_t)
    m.train_cnn()

    print '--------------- Evaluation on MapSwipe Samples ---------------'
    m.set_evaluation_input(img_X2, Y2)
    m.evaluate()
