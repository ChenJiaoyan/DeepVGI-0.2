#! /usr/bin/python

import sys
import os
import shutil
sys.path.append("../lib")
import FileIO

img_dir1 = '../data/image_project_922/'
img_dir2 = '../data/image_project_922_negative/'
outdir = '../../zhou/wrong_building_imgs/'
log_file = 'result_check_building.log'
building = '../data/buildings.csv'
buildings = FileIO.read_lines(building, 0)

imgs1 = os.listdir(img_dir1)
imgs2 = os.listdir(img_dir2)

wrong_building = []
with open(log_file, 'r') as f:
    next(f)
    for line in f:
        wrong_id = line.split(' ')[0]
        wrong_building.append(wrong_id)

for building in buildings:
    item = building.strip().split(',')
    osm_id = item[0]
    if osm_id in wrong_building:
        task_x = item[5]
        task_y = item[6]
        img = task_x + '-' + task_y + '.jpeg'
        if img in imgs1:
            shutil.copy(os.path.join(img_dir1, img), outdir)
        elif img in imgs2:
            shutil.copy(os.path.join(img_dir2, img), outdir)
        else:
            print '%s img not exist' % img







