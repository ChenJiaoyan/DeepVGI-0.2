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

    imagery = os.listdir('../data/imagery')

    print 'moving file for test/Positive'
    for img in p_imgs:
        if img in imagery:
            shutil.copyfile(os.path.join('../data/imagery', img),
                            os.path.join('../samples0/test/Positive/',
                                         img.strip().replace('_18.', '.').replace('_', '-')))
        else:
            print '%s does NOT have image file' % img

    for img in n_imgs:
        if img in imagery:
            shutil.copy(os.path.join('../data/imagery', img),
                        os.path.join('../samples0/test/Negative/',
                                     img.strip().replace('_18.', '.').replace('_', '-')))
        else:
            print '%s does NOT have image file' % img
