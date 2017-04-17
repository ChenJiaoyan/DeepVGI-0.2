#! /usr/bin/python

# x: decision score; y1: labor (percentage of imgs to volunteers); y2: overall accuracy

import sys
import numpy as np

sys.path.append("../lib")
import NN_Model
import FileIO
import MapSwipe


def mapswipe_label(img, ms_p_imgs):
    tmp = img.split('_')
    x, y = int(tmp[0]), int(tmp[1])
    if [x, y] in ms_p_imgs:
        return 1.0
    else:
        return 0.0


def my_eva(img_files, labels, imgs_p, imgs_n):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i, img in enumerate(img_files):
        label = labels[i]
        if img in imgs_p:
            if label == 1:
                TP += 1
            else:
                FN += 1
        elif img in imgs_n:
            if label == 1:
                FP += 1
            else:
                TN += 1
        else:
            print '%s does not has predicted label' % img
            sys.exit(0)
    p = float(TP) / float(TP + FP)
    r = float(TP) / float(TP + FN)
    a = float(TP + TN) / float(TP + TN + FP + FN)
    return p, r, a


if __name__ == '__main__':
    model_name = sys.argv[1]
    img_X, Y, img_files = FileIO.read_external_test_sample()
    imgs_p, imgs_n = FileIO.read_external_test_img()
    client = MapSwipe.MSClient()
    ms_p_imgs = client.read_p_images()
    print 'threshold, mapswipe_imgs, precision, recall, accuracy'
    x = np.arange(0, 0.51, 0.05)
    m = NN_Model.Model(img_X, Y, model_name)
    for x0 in x:
        scores, labels = m.predict()
        ms_index = np.where((scores >= x0) & (scores <= 1.0 - x0))
        for i in ms_index[0]:
            labels[i] = mapswipe_label(img_files[i], ms_p_imgs)
        ms_n = ms_index[0].shape[0]
        precision, recall, accuracy = my_eva(img_files, labels, imgs_p, imgs_n)
        print '%f, %d, %f, %f, %f' % (x0, ms_n, precision, recall, accuracy)
