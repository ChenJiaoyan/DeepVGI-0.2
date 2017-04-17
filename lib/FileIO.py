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
    lines = read_lines("../data/test_imgs.csv", 0)
    lines_p = read_lines("../data/test_positive_imgs.csv", 0)
    imgs_p, imgs_n = [], []
    for line in lines_p:
        imgs_p.append(line.strip())
    for line in lines:
        if line.strip() not in imgs_p:
            imgs_n.append(line.strip())
    n = len(imgs_p) + len(imgs_n)
    img_X = np.zeros((n, 256, 256, 3))
    label = np.zeros((n, 2))
    dir1 = '../data/imagery/'
    i = 0
    img_files = []
    for img in imgs_p:
        if os.path.exists(os.path.join(dir1, img)):
            img_X[i] = misc.imread(os.path.join(dir1, img))
            label[i, 1] = 1
            img_files.append(img)
            i += 1
    n_p = i
    print 'positive external testing samples: %d \n' % n_p
    for img in imgs_n:
        if os.path.exists(os.path.join(dir1, img)):
            img_X[i] = misc.imread(os.path.join(dir1, img))
            label[i, 0] = 1
            img_files.append(img)
            i += 1
    n_n = i - n_p
    print 'negative external testing samples: %d \n' % n_n
    return img_X[0:i], label[0:i], img_files
