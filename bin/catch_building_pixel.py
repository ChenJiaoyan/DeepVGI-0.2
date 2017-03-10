#! /usr/bin/python

import sys

sys.path.append("../lib")
import FileIO

import os
import random
from skimage import io
import matplotlib.pyplot as plt

L = 50
lines = FileIO.read_lines('../data/building_samples.csv', 1)
scp_pre = 'scp -i ~/.ssh/spatialdata_keypair.pem ubuntu@129.206.7.141:/home/ubuntu/DeepVGI-0.2/data/image_project_922/'
scp_suf = '../data/img_examples/'
img_name = ''
item = ''
results = []


def on_click(click):
    x, y = int(click.xdata), int(click.ydata)
    if x - L / 2 <= 0 or y - L / 2 <= 0 or x + L / 2 > 256 or y + L / 2 > 256:
        print '%d,%d ignored' % (x, y)
    else:
        print '%d,%d selected' % (x, y)
        results.append('%s,%d,%d\n' % (item, x, y))


for line in lines:
    item = line.strip()
    line_split = item.split(',')
    img_name = '%s-%s.jpeg' % (line_split[1], line_split[2])
    os.system('%s%s %s' % (scp_pre, img_name, scp_suf))
    print '%s: ' % img_name

    if 'yes' in line_split[0]:
        print 'OSM building'
        results.append(item + '\n')
        if random.random() <= 0.1:
            x, y = int(line_split[3]), int(line_split[4])
            print '%d,%d' % (x, y)
            img = io.imread('../data/img_examples/' + img_name)
            row, col = y, x
            img[row, col] = [255, 0, 0]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.imshow(img)
            plt.show()
    else:
        print 'NOT OSM building'
        img = io.imread('../data/img_examples/' + img_name)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(img)
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

print '%d buildings saved' % FileIO.save_lines('../data/building_samples.csv', results)
