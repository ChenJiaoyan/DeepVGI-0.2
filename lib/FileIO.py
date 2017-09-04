#! /usr/bin/python

import csv
import os
from scipy import misc
import numpy as np
import random
import MapSwipe


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


def get_urban_tasks(urban_file='../data/malawi_urban_extent.csv'):
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
        img_X[i + pn] = misc.imread(os.path.join('../samples0/test/Negative/', img))
        label[i + pn, 0] = 1

    return img_X, label


def read_valid_sample(n):
    client = MapSwipe.MSClient()
    MS_valid_p = client.MS_valid_positive()
    MS_valid_n = client.MS_valid_negative()

    print 'MS_valid_p: %d \n' % len(MS_valid_p)
    print 'MS_valid_n: %d \n' % len(MS_valid_n)

    if len(MS_valid_p) < n / 2 or len(MS_valid_n) < n / 2:
        print 'n is set too large; use all the samples for testing'
        n = len(MS_valid_p) * 2 if len(MS_valid_p) < len(MS_valid_n) else len(MS_valid_n) * 2

    img_X1, img_X0 = np.zeros((n / 2, 256, 256, 3)), np.zeros((n / 2, 256, 256, 3))
    MS_valid_p = random.sample(MS_valid_p, n / 2)
    for i, img in enumerate(MS_valid_p):
        img_X1[i] = misc.imread(os.path.join('../samples0/valid/MS_record/', img))

    MS_valid_n = random.sample(MS_valid_n, n / 2)
    for i, img in enumerate(MS_valid_n):
        img_X0[i] = misc.imread(os.path.join('../samples0/valid/MS_negative/', img))

    X = np.concatenate((img_X1[0:n / 2], img_X0[0:n / 2]))

    label = np.zeros((n, 2))
    label[0:n / 2, 1] = 1
    label[n / 2:n, 0] = 1

    return X, label


def osm_building_weight():
    task_w = {}
    osm_buildings = csv_reader("../data/buildings.csv")
    for row in osm_buildings:
        task_x = row['task_x']
        task_y = row['task_y']
        k = '%s-%s.jpeg' % (task_x, task_y)
        task_w[k] = 1
    return task_w

def read_urban_valid_sample(n):
    client = MapSwipe.Urban_client()
    urban_valid_p = client.valid_positive()
    urban_valid_n = client.valid_negative()

    print 'urban_valid_p: %d \n' % len(urban_valid_p)
    print 'urban_valid_n: %d \n' % len(urban_valid_n)

    if len(urban_valid_p) < n / 2 or len(urban_valid_n) < n / 2:
        print 'n is set too large; use all the samples for testing'
        n = len(urban_valid_p) * 2 if len(urban_valid_p) < len(urban_valid_n) else len(urban_valid_n) * 2

    img_X1, img_X0 = np.zeros((n / 2, 256, 256, 3)), np.zeros((n / 2, 256, 256, 3))
    urban_valid_p = random.sample(urban_valid_p, n / 2)
    for i, img in enumerate(urban_valid_p):
        img_X1[i] = misc.imread(os.path.join('../samples0/valid/MS_record/', img))

    urban_valid_n = random.sample(urban_valid_n, n / 2)
    for i, img in enumerate(urban_valid_n):
        img_X0[i] = misc.imread(os.path.join('../samples0/valid/MS_negative/', img))

    X = np.concatenate((img_X1[0:n / 2], img_X0[0:n / 2]))

    label = np.zeros((n, 2))
    label[0:n / 2, 1] = 1
    label[n / 2:n, 0] = 1

    return X, label