#! /usr/bin/python

import sys

sys.path.append("../lib")
import FileIO

import random
import numpy as np
from skimage import io
from skimage import exposure
from skimage import color
import matplotlib.pyplot as plt

hist_n = 128
PATCH = 20
n = 150


def col_hist(item):
    img_dir = '../data/img_examples/'
    tmp = item.split(',')
    task_x, task_y = int(tmp[1]), int(tmp[2])
    pixel_x, pixel_y = int(tmp[3]), int(tmp[4])
    img_f = '%s%d-%d.jpeg' % (img_dir, task_x, task_y)
    img = io.imread(img_f)
    img = color.rgb2gray(img)
    x1, x2 = pixel_x - PATCH / 2, pixel_x + PATCH / 2
    y1, y2 = pixel_y - PATCH / 2, pixel_y + PATCH / 2
    img_hist = exposure.histogram(img[x1:x2, y1:y2], hist_n)
    print img_hist
    return img_hist[0]


lines = FileIO.read_lines("../data/building_samples.csv", 1)
y_lines, n_lines = [], []
y_hist, n_hist = np.zeros((hist_n,)), np.zeros((hist_n,))
for line in lines:
    if 'yes' in line:
        y_lines.append(line)
    else:
        n_lines.append(line)

y_lines = random.sample(y_lines, n)
n_lines = random.sample(n_lines, n)
for i in range(n):
    y_hist += col_hist(y_lines[i])
    n_hist += col_hist(n_lines[i])

y_hist /= PATCH * PATCH * n
n_hist /= PATCH * PATCH * n
hist = (y_hist + n_hist) / 2
X = np.array(range(hist_n))

plt.plot(X, n_hist)
plt.plot(X, y_hist)
plt.show()
