#! /usr/bin/python

# x: decision score; y1: labor (percentage of imgs to volunteers); y2: overall accuracy

import sys
import numpy as np

sys.path.append("../lib")
import FileIO
import MapSwipe


def mapswipe_label(img, ms_p_imgs):
    tmp = img.split('_')
    x, y = int(tmp[0]), int(tmp[1])
    if [x, y] in ms_p_imgs:
        return 1.0
    else:
        return 0.0


def my_eva(imgs, labels, imgs_p, imgs_n):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i, img in enumerate(imgs):
        label = labels[i]
        if img in imgs_p:
            if label == 1:
                TP += 1
            else:
                FN += 1
        #        print '%d, %s: FN' % (i+1, img)
        elif img in imgs_n:
            if label == 1:
                FP += 1
        #        print '%d, %s: FP' % (i+1, img)
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
    lines = FileIO.read_lines("../data/test_scores.csv", 0)
    scores, labels = [], []
    for line in lines:
        tmp = line.strip().split(',')
        scores.append(float(tmp[0]))
        labels.append(float(tmp[1]))

    imgs_p, imgs_n = FileIO.read_external_test_img()
    imgs = imgs_p + imgs_n

    client = MapSwipe.MSClient()
    ms_p_imgs = client.read_p_images2()

    print 'threshold, mapswipe_imgs, precision, recall, accuracy'
    x = np.arange(0, 0.51, 0.05)
    for x0 in x:
        ms_n = 0
        for i, img in enumerate(imgs):
            score = scores[i]
            if 0.5 - x0 <= score <= 0.5 + x0:
                labels[i] = mapswipe_label(img, ms_p_imgs)
                ms_n += 1

        precision, recall, accuracy = my_eva(imgs, labels, imgs_p, imgs_n)
        print '%f, %d, %f, %f, %f' % (x0, ms_n, precision, recall, accuracy)
