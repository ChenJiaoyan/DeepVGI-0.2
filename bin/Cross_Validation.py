#! /usr/bin/python

import sys
sys.path.append("../lib")
import FileIO

import os
import random
import shutil

CV_i = 0
CV_n = 4
N = 40000


def cv(imgs):
    random.shuffle(imgs)
    imgs = imgs[0:N/2]
    l = len(imgs)
    b = l / CV_n
    valid = imgs[CV_i * b: (CV_i + 1) * b]
    train = imgs[0:CV_i * b] + imgs[(CV_i + 1) * b: l]
    return train, valid

if __name__ == '__main__':
    record_dir = '../data/image_project_922/'
    negative_dir = '../data/image_project_922_negative/'
    MS_records = os.listdir(record_dir)
    MS_negative = os.listdir(negative_dir)
    e_imgs = FileIO.read_lines("../data/test_imgs.csv", 0)
    for e_img in e_imgs:
        e_img = e_img.strip().replace("_18.", ".")
        if e_img in MS_records:
            MS_records.remove(e_img)
        if e_img in MS_negative:
            MS_records.remove(e_img)
    MS_records_train, MS_records_valid = cv(MS_records)
    MS_negative_train, MS_negative_valid = cv(MS_negative)

    for img in MS_records_train:
        shutil.copy(os.path.join(record_dir, img), '../samples0/train/MS_record/')

    for img in MS_negative_train:
        shutil.copy(os.path.join(record_dir, img), '../samples0/train/MS_negative/')

    for img in MS_records_valid:
        shutil.copy(os.path.join(record_dir, img), '../samples0/valid/MS_record/')

    for img in MS_negative_valid:
        shutil.copy(os.path.join(record_dir, img), '../samples0/valid/MS_negative/')
