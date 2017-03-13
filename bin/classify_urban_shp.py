#! /usr/bin/python

import sys
import csv
from osgeo import ogr

in_file = "../data/shp/project_922.shp"
out_file = "../data/malawi_urban.csv"
shp_file = "../data/shp/malawi_urban.shp"


def get_Layer(shp_name):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    source = driver.Open(shp_name, 0)
    layer = source.GetLayer()
    #layer_defn = layer.GetLayerDefn()
    return layer

all_922 = []
all_urban = []

for feature1 in get_Layer(in_file):
    geometry1 = feature1.GetGeometryRef()
    all_922.append(geometry1.Clone())

for feature2 in get_Layer(shp_file):
    geometry2 = feature2.GetGeometryRef()
    all_urban.append(geometry2.Clone())

f = open(out_file, 'w')
f.write('task_x' + ',' + 'task_y' + ',' + 'classification' + '\n')

for urban in all_urban:
    for tile in all_922:
        for feature in get_Layer(in_file):
            if urban.Intersect(tile) == True :
                task_x = feature.GetField("task_x")
                task_y = feature.GetField("task_y")
                item = task_x + ',' + task_y + ',' + 'urban' + '\n'
                f.write(item)
            else:
                task_x = feature.GetField("task_x")
                task_y = feature.GetField("task_y")
                item = task_x + ',' + task_y + ',' + 'rural' + '\n'
                f.write(item)

f.close()
