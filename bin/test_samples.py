#! /usr/bin/python

import sys

sys.path.append("../lib")
import FileIO

import shutil
import os

if __name__ == '__main__':
    a_imgs = FileIO.read_lines("../data/test_imgs.csv", 0)
    p_imgs = FileIO.read_lines("../data/test_positive_imgs.csv", 0)
    n_imgs = list(set(a_imgs).difference(set(p_imgs)))

    p_imgs = [img.replace('-18.', '.') for img in p_imgs]
    n_imgs = [img.replace('-18.', '.') for img in n_imgs]

    record = os.listdir('../data/image_project_922')
    negative = os.listdir('../data/image_project_922_negative')

    print 'moving file for test/Positive'
    for img in p_imgs:
        if img in record:
            shutil.copy(os.path.join('../data/image_project_922', img), '../samples0/test/Positive/')
        elif img in negative:
            shutil.copy(os.path.join('../data/image_project_922_negative', img), '../samples0/test/Positive/')
        else:
            print '%s does NOT have image file' % img

    for img in n_imgs:
        if img in record:
            shutil.copy(os.path.join('../data/image_project_922', img), '../samples0/test/Negative/')
        elif img in negative:
            shutil.copy(os.path.join('../data/image_project_922_negative', img), '../samples0/test/Negative/')
        else:
            print '%s does NOT have image file' % img
