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

n1, n0 = 2500, 2500
img_X1, img_X0 = np.zeros((n1, 256, 256, 3)), np.zeros((n0, 256, 256, 3))
label = np.zeros((n1 + n0, 2))
label[0:n1, 1] = 1
label[n1:(n1 + n0), 0] = 1

osm_buildings = FileIO.read_lines("../data/buildings.csv", 1)
img_dir = '../data/image_project_922/'
imgs = os.listdir(img_dir)
osm_imgs, none_osm_imgs = [], []
for img in imgs:
    i1, i2 = img.index('-'), img.index(',')
    task_x, task_y = img[0:i1], img[(i1 + 1):i2]
    if task_x + ',' + task_y in imgs:
        osm_imgs.append(img)
    else:
        none_osm_imgs.append(img)
osm_imgs = random.sample(osm_imgs, n1)
none_osm_imgs = random.sample(none_osm_imgs, n0)

for i, img in enumerate(osm_imgs):
    x1 = misc.imread(img)
    img_X1[i] = x1
for i, img in enumerate(none_osm_imgs):
    x0 = misc.imread(img)
    img_X0[i] = x0

img_X = np.concatenate((img_X1, img_X0))
m = NN_Model.Model(img_X, label,'')
m.train_cnn()

