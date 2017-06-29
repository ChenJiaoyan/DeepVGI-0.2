#! /usr/bin/python

import sys
import csv
import numpy as np
from osgeo import ogr

in_file = "../data/all_tasks_922.csv"
out_file = "../data/malawi_urban_extent.csv"
shp_file = "../data/shp/malawi_urban_extent.shp"

task_xy = []
task_geom = []
with open(in_file) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['project_id'] == '922':
            task_x = row['task_id'].strip().split('-')[1]
            task_y = row['task_id'].strip().split('-')[2]
            wkt = row['task_geom']
            poly = ogr.CreateGeometryFromWkt(wkt)
            task_xy.append(str(task_x) + ',' + str(task_y))
            task_geom.append(poly)

driver = ogr.GetDriverByName("ESRI Shapefile")
source = driver.Open(shp_file, 0)
layer = source.GetLayer()
all_urban = []
for feature in layer:
    geometry = feature.GetGeometryRef()
    all_urban.append(geometry.Clone())

f = open(out_file, 'w')
f.write('task_x' + ',' + 'task_y' + ',' + 'classification' + '\n')
m = 0
n = 0
for urban in all_urban:
    for i, geom in enumerate(task_geom):
        if urban.Intersect(geom):
            line = task_xy[i] + ',' + 'urban_extent' + '\n'
            m += 1
            f.write(line)
        else:
            line = task_xy[i] + ',' + 'other' + '\n'
            n += 1
            f.write(line)

f.close()
source.Destroy()
print ("Total urban_extent tile: %s \nTotal rural tiles: %s" % (m, n))