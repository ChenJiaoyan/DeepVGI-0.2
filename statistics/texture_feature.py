#! /usr/bin/python

import sys

sys.path.append("../lib")
import FileIO

import random
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color
from skimage.feature import greycomatrix, greycoprops

PATCH = 21
n = 10


def glcm_xy(item):
    img_dir = '../data/img_examples/'
    tmp = item.split(',')
    task_x, task_y = int(tmp[1]), int(tmp[2])
    pixel_x, pixel_y = int(tmp[3]), int(tmp[4])
    img_f = '%s%d-%d.jpeg' % (img_dir, task_x, task_y)
    img = color.rgb2gray(io.imread(img_f))
    x1, x2 = pixel_x - PATCH / 2, pixel_x + PATCH / 2
    y1, y2 = pixel_y - PATCH / 2, pixel_y + PATCH / 2
    glcm = greycomatrix(img[x1:x2, y1:y2], [5], [0], 256, symmetric=True, normed=True)
    x = greycoprops(glcm, 'dissimilarity')[0, 0]
    y = greycoprops(glcm, 'correlation')[0, 0]
    print '%s  %f %f \n' % (item, x, y)
    return x, y


lines = FileIO.read_lines("../data/building_samples.csv", 1)
y_lines, n_lines = [], []
for line in lines:
    if 'yes' in line:
        y_lines.append(line)
    else:
        n_lines.append(line)

y_lines = random.sample(y_lines, n)
n_lines = random.sample(n_lines, n)
y_glcm = np.zeros((n, 2), dtype=np.float32)
n_glcm = np.zeros((n, 2), dtype=np.float32)

for i in range(n):
    y_glcm[i, 0], y_glcm[i, 1] = glcm_xy(y_lines[i])
    n_glcm[i, 0], n_glcm[i, 1] = glcm_xy(n_lines[i])

plt.plot(y_glcm[:, 0], y_glcm[:, 1], 'go')
plt.plot(n_glcm[:, 0], n_glcm[:, 1], 'bo')
plt.show()
