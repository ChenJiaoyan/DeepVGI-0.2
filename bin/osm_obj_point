#! /usr/bin/python

import sys

sys.path.append("../lib")
import MapSwipe
import FileIO
from osgeo import ogr

shp_file = "../data/shp/buildings.shp"
results = ['osm_id,name,type,lat,lon,task_x,task_y,pixel_x,pixel_y\n']
out_file = "../data/buildings.csv"

driver = ogr.GetDriverByName("ESRI Shapefile")
source = driver.Open(shp_file, 0)
layer = source.GetLayer()
obj_count = layer.GetFeatureCount()
print 'obj count: %d' % obj_count
for i in range(0, obj_count):
    obj = layer.GetFeature(i)
    osm_id = obj.GetField('osm_id')
    name = str(obj.GetField('name'))
    typ = str(obj.GetField('type'))
    geom = obj.GetGeometryRef()
    centroid = geom.Centroid()
    lon = centroid.GetX()
    lat = centroid.GetY()
    task_x, task_y, pixel_x, pixel_y = MapSwipe.cal_pixel(lat, lon)
    s = str(osm_id) + ',' + str(name.replace("'", " ").replace(",", ".")) + ',' + str(typ) + ',' + str(lat) + ',' + str(
        lon) + ',' + str(task_x) + ',' + str(task_y) + ',' + str(pixel_x) + ',' + str(pixel_y) + '\n'
    print s
    results.append(s)

FileIO.save_lines(out_file, results)
