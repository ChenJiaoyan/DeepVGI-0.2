#! /usr/bin/python

import csv
import os
from scipy import misc
import numpy as np


def csv_reader(file_name):
    cf = open(file_name)
    reader = csv.DictReader(cf)
    return reader


def read_lines(file_name, start_line):
    f = open(file_name)
    lines = f.readlines()
    f.close()
    return lines[start_line:]


def save_lines(file_name, lines):
    f = open(file_name, 'w')
    f.writelines(lines)
    f.close()
    return len(lines)


def get_urban_tasks(urban_file='../data/malawi_urban.csv'):
    urbans = []
    for row in csv_reader(urban_file):
        task_x = row['task_x']
        task_y = row['task_y']
        c = row['classification']
        if c == 'urban':
            urbans.append(task_x + ',' + task_y)
    return urbans


def read_external_test_img():
    lines = read_lines("../data/test_imgs.csv", 0)
    lines_p = read_lines("../data/test_positive_imgs.csv", 0)
    imgs_p, imgs_n = [], []
    for line in lines_p:
        imgs_p.append(line.strip())
    for line in lines:
        if line.strip() not in imgs_p:
            imgs_n.append(line.strip())
    return imgs_p, imgs_n


def read_external_test_sample():
    p_imgs = os.listdir('../samples0/test/Positive')
    pn = len(p_imgs)
    n_imgs = os.listdir('../samples0/test/Negative')
    nn = len(n_imgs)

    img_X = np.zeros((pn + nn, 256, 256, 3))
    label = np.zeros((pn + nn, 2))

    for i, img in enumerate(p_imgs):
        img_X[i] = misc.imread(os.path.join('../samples0/test/Positive/', img))
        label[i, 1] = 1

    for i, img in enumerate(n_imgs):
        img_X[i+pn] = misc.imread(os.path.join('../samples0/test/Negative/', img))
        label[i+pn, 0] = 1

    return img_X, label


def osm_building_weight():
    task_w = {}
    osm_buildings = FileIO.csv_reader("../data/buildings.csv")
    for row in osm_buildings:
        task_x = row['task_x']
        task_y = row['task_y']
        k = '%s-%s' % (task_x, task_y)
        task_w[k] = 1
    return task_w
