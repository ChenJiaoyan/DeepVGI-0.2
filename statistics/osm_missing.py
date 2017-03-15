#! /usr/bin/python

# Calculate the missing rate

import sys
import random

sys.path.append("../lib")
import MapSwipe
import FileIO

client = MapSwipe.MSClient(922, 'Malawi')

matched = []
missed = []

osm_lines = FileIO.read_lines("../data/buildings.csv", 1)
osm_imgs = []
for line in osm_lines:
    tmp = line.strip().split(',')
    osm_imgs.append(tmp[5] + ',' + tmp[6])
osm_imgs = sorted(osm_imgs)

p_imgs = client.read_p_images()
for i in p_imgs:
    p_img = str(i[0]) + ',' + str(i[1])
    if p_img in osm_imgs:
        print '%s matched by OSM' % p_img
        matched.append(p_img)
    else:
        print '%s missed by OSM' % p_img
        missed.append(p_img)

print '-----------------------random images start----------------------------'
n = 200
samples = []
L = 50

matched_r = random.sample(matched, n)
for img in matched_r:
    for line in osm_lines:
        if img in line:
            tmp = line.strip().split(',')
            x, y = int(tmp[-2]), int(tmp[-1])
            if x - L / 2 <= 0 or y - L / 2 <= 0 or x + L / 2 > 256 or y + L / 2 > 256:
                continue
            else:
                s = 'yes,%s,%d,%d\n' % (img, x, y)
                samples.append(s)

missed_r = random.sample(missed, n)
for img in missed_r:
    s = 'no,%s\n' % img
    samples.append(s)

print '%d random samples' % FileIO.save_lines("../data/building_samples.csv", samples)
print '-----------------------random images end----------------------------'

print '%d out of %d images are missed by OSM' % (len(missed), len(p_imgs))
print 'missing rate: %f' % float(len(missed) / float(len(p_imgs)))
