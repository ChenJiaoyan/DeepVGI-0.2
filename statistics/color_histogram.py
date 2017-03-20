#! /usr/bin/python

import sys

sys.path.append("../lib")
import FileIO

import random
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import exposure

hist_n = 256
PATCH = 16
n1, n2 = 9, 9
n = n1 * n2


def crop(item):
    img_dir = '../data/img_examples/'
    tmp = item.split(',')
    task_x, task_y = int(tmp[1]), int(tmp[2])
    pixel_x, pixel_y = int(tmp[3]), int(tmp[4])
    img_f = '%s%d-%d.jpeg' % (img_dir, task_x, task_y)
    img = io.imread(img_f)
    x1, x2 = pixel_x - PATCH / 2, pixel_x + PATCH / 2
    y1, y2 = pixel_y - PATCH / 2, pixel_y + PATCH / 2
    return img[x1:x2, y1:y2]


def hexencode(rgb):
    return '#%02x%02x%02x' % rgb


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
y_img = np.zeros((n1 * PATCH, n2 * PATCH, 3), dtype=np.uint8)
n_img = np.zeros((n2 * PATCH, n2 * PATCH, 3), dtype=np.uint8)

for i in range(n1):
    for j in range(n2):
        y_crop = crop(y_lines[i * n1 + j])
        y_img[i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH] = y_crop
        n_crop = crop(n_lines[i * n1 + j])
        n_img[i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH] = n_crop


rgb = ['r', 'g', 'b']

for i, c in enumerate(rgb):
    y_hist = exposure.histogram(y_img[:, :, i])
    n_hist = exposure.histogram(n_img[:, :, i])
    #plt.plot(y_hist[1], y_hist[0] / float(y_img.shape[0] * y_img.shape[1]), c)
    plt.plot(n_hist[1], n_hist[0] / float(n_img.shape[0] * n_img.shape[1]), c)
    plt.xlim(0, 255)
    plt.ylim(0, 0.03)
    plt.title("RGB Color Histogram")

plt.show()

io.imsave('../data/img_examples/y_crops.jpeg', y_img)
io.imsave('../data/img_examples/n_crops.jpeg', n_img)
