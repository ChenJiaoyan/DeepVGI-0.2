#! /usr/bin/python

#test

import sys

sys.path.append("../lib")
import MapSwipe
import FileIO

client = MapSwipe.MSClient(922, 'Malawi')

matched = []
missed = []

osm_lines = FileIO.read_lines("../data/buildings.csv", 1)
osm_tasks = []
for line in osm_lines:
    tmp = line.strip().split(',')
    osm_tasks.append(tmp[5] + '_' + tmp[6])
osm_tasks = sorted(osm_tasks)

p_imgs = client.read_p_images()
for i in p_imgs:
    p_img = str(i[0]) + '_' + str(i[1])
    if p_img in osm_tasks:
        print '%s matched by OSM' % p_img
        matched.append(p_img)
    else:
        print '%s missed by OSM' % p_img
        missed.append(p_img)

print '%d out of %d images are missed by OSM' % (len(missed), len(p_imgs))
print 'missing rate: %f' % float(len(missed)/float(len(p_imgs)))
