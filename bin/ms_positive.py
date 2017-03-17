#! /usr/bin/python

from osgeo import ogr
import osgeo.osr as osr

msfile = "../data/shp/project_922.shp"
driver = ogr.GetDriverByName("ESRI Shapefile")
source = driver.Open(msfile, 0)
layer = source.GetLayer()

srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

outshp = '../data/shp/ms_positive.shp'
outdriver = ogr.GetDriverByName("ESRI Shapefile")
outsource = outdriver.CreateDataSource(outshp)
outlayer = outsource.CreateLayer("malawi_positive", srs, geom_type=ogr.wkbMultiPolygon)

inLayerDefn = layer.GetLayerDefn()
for i in range(0, inLayerDefn.GetFieldCount()):
    fieldDefn = inLayerDefn.GetFieldDefn(i)
    outlayer.CreateField(fieldDefn)
outLayerDefn = outlayer.GetLayerDefn()

for i in range(0, layer.GetFeatureCount()):
    feature = layer.GetFeature(i)
    yes_count = feature.GetField("yes")
    maybe_count = feature.GetField("maybe")
    bad_img_count = feature.GetField("bad_image")
    if int(bad_img_count) == 0 and (int(yes_count) >= 2 or (int(maybe_count) + int(yes_count)) >= 3):
        outFeature = feature.Clone()
        #outFeature = ogr.Feature(outLayerDefn)
        for n in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(n).GetNameRef(), feature.GetField(n))
        outlayer.CreateFeature(outFeature)
        outFeature.Destroy()

source.Destroy()
outsource.Destroy()
