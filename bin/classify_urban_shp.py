#! /usr/bin/python

import sys
import csv
import numpy as np
from osgeo import ogr

in_file = "../data/shp/project_922.shp"
out_file = "../data/malawi_urban.csv"
shp_file = "../data/shp/malawi_urban.shp"

driver = ogr.GetDriverByName("ESRI Shapefile")
source1 = driver.Open(in_file, 0)
layer1 = source1.GetLayer()
source2 = driver.Open(shp_file, 0)
layer2 = source2.GetLayer()

all_922 = []
all_urban = []
item = []
for feature1 in layer1:
    geometry = feature1.GetGeometryRef()
    all_922.append(geometry.Clone())

    task_x = feature1.GetField("task_x")
    task_y = feature1.GetField("task_y")
    xy = str(task_x) + ',' + str(task_y)
    item.append(xy)
matrix = np.array((all_922, item))

for feature2 in layer2:
    geometry = feature2.GetGeometryRef()
    all_urban.append(geometry.Clone())

f = open(out_file, 'w')
f.write('task_x' + ',' + 'task_y' + ',' + 'classification' + '\n')
m = 0
n = 0
for urban in all_urban:
    for i in range(len(all_922)):
        if urban.Intersect(matrix[0][i]):
            line = matrix[1][i] + ',' + 'urban' + '\n'
            m += 1
            f.write(line)
        else:
            line = matrix[1][i] + ',' + 'rural' + '\n'
            n += 1
            f.write(line)

f.close()
source1.Destroy()
source2.Destroy()

print ("Total urban tile: %s \nTotal rural tiles: %s" % (m, n))