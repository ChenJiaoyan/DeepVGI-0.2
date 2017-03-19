#! /usr/bin/python

from osgeo import ogr
import osgeo.osr as osr

msfile = "../data/shp/project_922.shp"
driver = ogr.GetDriverByName("ESRI Shapefile")
source = driver.Open(msfile, 0)
layer = source.GetLayer()

srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

outshp = '../data/shp/ms_negative.shp'
outdriver = ogr.GetDriverByName("ESRI Shapefile")
outsource = outdriver.CreateDataSource(outshp)
outlayer = outsource.CreateLayer("malawi_negative", srs, geom_type=ogr.wkbMultiPolygon)

inLayerDefn = layer.GetLayerDefn()
for i in range(0, inLayerDefn.GetFieldCount()):
    fieldDefn = inLayerDefn.GetFieldDefn(i)
    outlayer.CreateField(fieldDefn)
outLayerDefn = outlayer.GetLayerDefn()

for i in range(0, layer.GetFeatureCount()):
    if i % 2000 == 0:
        print 'i == %d' % i
    feature = layer.GetFeature(i)
    yes_count = feature.GetField("yes")
    maybe_count = feature.GetField("maybe")
    bad_img_count = feature.GetField("bad_image")
    if int(bad_img_count) == 0 and int(yes_count) == 0 and int(maybe_count) == 0:
        outFeature = feature.Clone()
        for n in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(n).GetNameRef(), feature.GetField(n))
        outlayer.CreateFeature(outFeature)
        outFeature.Destroy()

source.Destroy()
outsource.Destroy()
