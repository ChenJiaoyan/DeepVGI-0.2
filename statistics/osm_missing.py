#! /usr/bin/python

# Calculate the overall/urban/rural missing rate

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


print '-----------------------overall missing rate start----------------------------'
p_imgs = client.read_p_images()
for i in p_imgs:
    p_img = str(i[0]) + ',' + str(i[1])
    if p_img in osm_imgs:
        print '%s matched by OSM' % p_img
        matched.append(p_img)
    else:
        print '%s missed by OSM' % p_img
        missed.append(p_img)
print '%d out of %d MapSwipe images are missed by OSM' % (len(missed), len(p_imgs))
print 'overall missing rate: %f\n' % float(len(missed) / float(len(p_imgs)))
print '-----------------------overall missing rate start----------------------------'


RURAL_URBAN_MISSING_RATE = True  # set to true if urban and rural missing rates need to be calculated
if not RURAL_URBAN_MISSING_RATE:
    sys.exit(0)

print '-----------------------urban/rural missing rate start----------------------------'
urban_tasks = FileIO.get_urban_tasks()
u_match_num, u_miss_num = 0, 0
r_match_num, r_miss_num = 0, 0
for i in p_imgs:
    p_img = str(i[0]) + ',' + str(i[1])
    if p_img in urban_tasks:
        if p_img in osm_imgs:
            u_match_num += 1
        else:
            u_miss_num += 1
    else:
        if p_img in osm_imgs:
            r_match_num += 1
        else:
            r_miss_num += 1

print '%d out of %d MapSwipe urban images are missed by OSM' % (u_miss_num, (u_miss_num + u_match_num))
print 'urban missing rate: %f\n' % (float(u_miss_num) / float(u_miss_num + u_match_num))

print '%d out of %d MapSwipe rural images are missed by OSM' % (r_miss_num, (r_miss_num + r_match_num))
print 'rural missing rate: %f\n' % (float(r_miss_num) / float(r_miss_num + r_match_num))

print '-----------------------urban/rural missing rate end----------------------------'


SAMPLING = False    # set to true if the following sampling codes need to be executed
if not SAMPLING:
    sys.exit(0)

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
