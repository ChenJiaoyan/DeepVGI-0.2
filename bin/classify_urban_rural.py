#! /usr/bin/python

import sys
import csv
sys.path.append("../lib")
import MapSwipe
from osgeo import ogr

in_file = "../data/project_922.csv"
out_file = "../data/malawi_urban.csv"
shp_file = "../data/shp/malawi_urban.shp"

driver = ogr.GetDriverByName("ESRI Shapefile")
source = driver.Open(shp_file, 0)
layer = source.GetLayer()
all_geoms = []
for feature in layer:
	geometry = feature.GetGeometryRef()
	all_geoms.append(geometry.Clone())

f = open(out_file, 'w')
f.write('task_x' + ',' + 'task_y' + ',' + 'classification' + '\n')

csvfile = open(in_file)
reader = csv.DictReader(csvfile)
m = 0
n = 0
for row in reader:
    task_x = float(row['task_x'])
    task_y = float(row['task_y'])
    lat_top, lon_left, lat_bottom, lon_right = MapSwipe.cal_lat_lon(task_x, task_y)

    pt_lt = ogr.Geometry(ogr.wkbPoint)
    pt_lt.AddPoint(lon_left, lat_top)

    pt_lb = ogr.Geometry(ogr.wkbPoint)
    pt_lb.AddPoint(lon_left, lat_bottom)

    pt_rt = ogr.Geometry(ogr.wkbPoint)
    pt_rt.AddPoint(lon_right, lat_top)

    pt_rb = ogr.Geometry(ogr.wkbPoint)
    pt_rb.AddPoint(lon_right, lat_bottom)
    for poly in all_geoms:
        if pt_lt.Distance(poly) >= 0.0 and pt_lb.Distance(poly) >= 0.0 and \
                        pt_rt.Distance(poly) >= 0.0 and pt_rb.Distance(poly) >= 0.0:
            item = str(task_x) + ',' + str(task_y) + ',' + 'rural' + '\n'
            f.write(item)
            n += 1
            #break
        else:
            item = str(task_x) + ',' + str(task_y) + ',' + 'urban' + '\n'
            f.write(item)
            m += 1
f.close()
feature.Destroy()
source.Destroy()
print ("Total urban tile: %s \n Total rural tiles: %s" % (m, n))

