#! /usr/bin/python

import sys

sys.path.append("../lib")
import FileIO

import os
import random
import shutil

CV_i = 0
CV_n = 4
Record_N = 160000
Negative_N = 20000


def cv(imgs, N):
    random.shuffle(imgs)
    imgs = imgs[0:N]
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
    test_imgs = os.listdir('../samples0/test/Positive') + os.listdir('../samples0/test/Negative')
    for e_img in test_imgs:
        if e_img in MS_records:
            MS_records.remove(e_img)
        if e_img in MS_negative:
            MS_records.remove(e_img)
    MS_records_train, MS_records_valid = cv(MS_records, Record_N)
    MS_negative_train, MS_negative_valid = cv(MS_negative, Negative_N)

    print 'moving file for train/MS_record'
    for img in MS_records_train:
        shutil.copy(os.path.join(record_dir, img), '../samples0/train/MS_record/')

    print 'moving file for train/MS_negative'
    for img in MS_negative_train:
        shutil.copy(os.path.join(negative_dir, img), '../samples0/train/MS_negative/')

    print 'moving file for valid/MS_record'
    for img in MS_records_valid:
        shutil.copy(os.path.join(record_dir, img), '../samples0/valid/MS_record/')

    print 'moving file for valid/MS_negative'
    for img in MS_negative_valid:
        shutil.copy(os.path.join(negative_dir, img), '../samples0/valid/MS_negative/')
