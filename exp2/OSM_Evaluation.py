#! /usr/bin/python

# (1) train CNN with OSM labeled images; (2) evaluate the CNN with testing MapSwipe images
import os
import sys
import random
import numpy as np
from scipy import misc

sys.path.append("../lib")
import NN_Model
import FileIO

n1, n0 = 50, 50


def osm_building_weight():
    task_w = {}
    osm_buildings = FileIO.csv_reader("../data/buildings.csv")
    for row in osm_buildings:
        task_x = row['task_x']
        task_y = row['task_y']
        k = '%s-%s' % (task_x, task_y)
        task_w[k] = 1
    return task_w


def read_sample():
    img_X1, img_X0 = np.zeros((n1, 256, 256, 3)), np.zeros((n0, 256, 256, 3))
    label = np.zeros((n1 + n0, 2))
    label[0:n1, 1] = 1
    label[n1:(n1 + n0), 0] = 1

    task_w = osm_building_weight();
    img_dir = '../data/image_project_922/'
    imgs = os.listdir(img_dir)
    osm_imgs, none_osm_imgs = [], []
    for img in imgs:
        i1, i2 = img.index('-'), img.index('.')
        task_x, task_y = img[0:i1], img[(i1 + 1):i2]
        k = '%s-%s' % (task_x, task_y)
        if task_w.has_key(k):
            osm_imgs.append(img)
        else:
            none_osm_imgs.append(img)
    osm_imgs = random.sample(osm_imgs, n1)
    none_osm_imgs = random.sample(none_osm_imgs, n0)

    for i, img in enumerate(osm_imgs):
        x1 = misc.imread(os.path.join(img_dir, img))
        img_X1[i] = x1
    for i, img in enumerate(none_osm_imgs):
        x0 = misc.imread(os.path.join(img_dir, img))
        img_X0[i] = x0

    return np.concatenate((img_X1, img_X0)), label

print '--------------- Read Samples ---------------'
img_X, Y = read_sample()
m = NN_Model.Model(img_X, Y, 'CNN_JY')
m.train_cnn()
