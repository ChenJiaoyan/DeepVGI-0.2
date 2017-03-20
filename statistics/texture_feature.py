#! /usr/bin/python

# present glcm texture feature

import sys

sys.path.append("../lib")
import FileIO

import random
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.feature import greycomatrix, greycoprops

PATCH = 32
n = 10


def glcm_xy(item):
    img_dir = '../data/img_examples/'
    tmp = item.strip().split(',')
    task_x, task_y = int(tmp[1]), int(tmp[2])
    pixel_x, pixel_y = int(tmp[3]), int(tmp[4])
    img_f = '%s%d-%d.jpeg' % (img_dir, task_x, task_y)
    img = io.imread(img_f)
    x1, x2 = pixel_x - PATCH / 2, pixel_x + PATCH / 2
    y1, y2 = pixel_y - PATCH / 2, pixel_y + PATCH / 2
    img_crop = img[y1:y2, x1:x2, 0]
    glcm = greycomatrix(img_crop, [5], [0], 256, symmetric=True, normed=True)
    x = greycoprops(glcm, 'correlation')[0, 0]
    y = greycoprops(glcm, 'dissimilarity')[0, 0]
    print '%s %d-%d.jpeg:  %f %f ' % (tmp[0], task_x, task_y, x, y)
    io.imsave('/tmp/%s_%d_%d_%d_%d.jpeg' % (tmp[0], task_x, task_y, pixel_x, pixel_y), img[y1:y2, x1:x2])
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

# use specific buildings
y_lines = ['yes,156913,142807,185,218',
           'yes,156913,142807,221,105',
           'yes,157020,142876,197,52',
           'yes,156524,143268,58,76',
           'yes,156818,142865,197,213',
           'yes,157109,142929,31,149']
n_lines = ['no,156588,142767,159,68',
           'no,156672,142698,63,118',
           'no,157001,142713,121,32',
           'no,156752,142809,108,124',
           'no,157022,142741,94,133',
           'no,156510,142971,26,163']
n = len(y_lines)

y_glcm = np.zeros((n, 2), dtype=np.float32)
n_glcm = np.zeros((n, 2), dtype=np.float32)

for i in range(n):
    y_glcm[i, 0], y_glcm[i, 1] = glcm_xy(y_lines[i])
    n_glcm[i, 0], n_glcm[i, 1] = glcm_xy(n_lines[i])

plt.plot(y_glcm[:, 0], y_glcm[:, 1], 'go', ms=11)
#plt.plot(n_glcm[:, 0], n_glcm[:, 1], 'bo', ms=11)
line_segs = 50
line_x = np.arange(0.016, 0.395, (0.368-0.016)/line_segs)
line_y = np.arange(10.651, 33.5, (32-10.651)/line_segs)
plt.plot(line_x, line_y, 'r--')
plt.xlim(-0.1, 0.5)
plt.ylim(10, 35)
plt.xlabel('GLCM Correlation')
plt.ylabel('GLCM Dissimilarity')

plt.show()
