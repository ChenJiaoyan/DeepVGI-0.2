#! /usr/bin/python

# (1) train CNN with OSM labeled images; (2) evaluate the CNN with testing MapSwipe images
import os
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
    te_p, te_n = 0, 0
    for img in test_imgs:
        if i >= n:
            break
        img_X[i] = misc.imread(os.path.join(img_dir, img))
        i1, i2 = img.index('-'), img.index('.')
        task_x, task_y = img[0:i1], img[(i1 + 1):i2]
        if [int(task_x), int(task_y)] in ms_p_imgs:
            label[i, 1] = 1
            i += 1
            te_p += 1
        elif [int(task_x), int(task_y)] in ms_n_imgs:
            label[i, 0] = 1
            i += 1
            te_n += 1
    print 'positive testing samples: %d \n ' % te_p
    print 'negative testing samples: %d \n ' % te_n
    return img_X, label


def read_train_sample(n1, n0, train_imgs, ms_n_imgs):
    img_X1, img_X0 = np.zeros((n1, 256, 256, 3)), np.zeros((n0, 256, 256, 3))
    label = np.zeros((n1 + n0, 2))

    task_w = osm_building_weight();
    img_dir1 = '../data/image_project_922/'
    img_dir2 = '../data/image_project_922_negative/'
    p_imgs, n_imgs = [], []

    for img in train_imgs:
        i1, i2 = img.index('-'), img.index('.')
        task_x, task_y = img[0:i1], img[(i1 + 1):i2]
        k = '%s-%s' % (task_x, task_y)
        if task_w.has_key(k):
            p_imgs.append(img)
        elif [int(task_x), int(task_y)] in ms_n_imgs:
            n_imgs.append(img)

    print 'p_imgs labeled by OSM: %d \n' % len(p_imgs)
    print 'n_imgs labeled by MS: %d \n' % len(n_imgs)

    p_imgs = random.sample(p_imgs, n1)
    for i, img in enumerate(p_imgs):
        if os.path.exists(os.path.join(img_dir1,img)):
            img_X1[i] = misc.imread(os.path.join(img_dir1, img))
        else:
            img_X1[i] = misc.imread(os.path.join(img_dir2, img))
    label[0:n1, 1] = 1

    n_imgs = random.sample(n_imgs, n0)
    for i, img in enumerate(n_imgs):
        img_X0[i] = misc.imread(os.path.join(img_dir2, img))
    label[n1:(n1 + n0), 0] = 1

    j = range(n1 + n0)
    random.shuffle(j)
    X = np.concatenate((img_X1, img_X0))
    return X[j], label[j]


if __name__ == '__main__':
    evaluate_only, tr_n1, tr_n0, tr_b, tr_e, tr_t, cv_i, te_n, nn = Parameters.deal_args(sys.argv[1:])
    cv_n = 4

    print '--------------- Read Samples ---------------'
    client = MapSwipe.MSClient()
    train_imgs, test_imgs = client.imgs_cross_validation(cv_i, cv_n)
    ms_p_imgs = client.read_p_images()
    ms_n_imgs = client.read_n_images()
    img_X, Y = read_train_sample(tr_n1, tr_n0, train_imgs, ms_n_imgs)
    m = NN_Model.Model(img_X, Y, nn + '_JY')

    if not evaluate_only:
        print '--------------- Training on OSM Labels---------------'
        m.set_batch_size(tr_b)
        m.set_epoch_num(tr_e)
        m.set_thread_num(tr_t)
        m.train(nn)
        print '--------------- Evaluation on Training Samples ---------------'
        m.evaluate()
    del img_X, Y, train_imgs
    gc.collect()

    print '--------------- Evaluation on MapSwipe Samples ---------------'
    img_X2, Y2 = read_test_sample(te_n, test_imgs, ms_p_imgs, ms_n_imgs)
    m.set_evaluation_input(img_X2, Y2)
    m.evaluate()
