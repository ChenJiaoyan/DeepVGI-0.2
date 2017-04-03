#! /usr/bin/python

import os, sys
import math
import csv
import urllib

class Tile:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

def tile_address(x, y):
    t = Tile()
    t.x = int(x)
    t.y = int(y)
    print"\nThe tile coordinates are x = {} and y = {}".format(t.x, t.y)
    return t

def tile_coords_and_zoom_to_quadKey(x, y, zoom):
    quadKey = ''
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        quadKey += str(digit)
    print "\nThe quadkey is {}".format(quadKey)
    return quadKey

def quadKey_to_url(quadKey):
    try:
        f = open('cfg/api_key.txt')
        api_key = f.read()
    except:
        print ("Something is wrong with your API key.\n"
               "Do you even have an API key?")

    # TODO get this into a config file, and set up others (Google, OSM, etc)
    tile_url = ("http://a0.ortho.tiles.virtualearth.net/tiles/a{}.jpeg?"
                "g=854&mkt=en-US&token={}".format(quadKey, api_key))
    print "\nThe tile URL is: {}".format(tile_url)
    return tile_url

def tile_to_quadKey(x, y, zoom):
    tile = tile_address(x, y)
    quadKey = tile_coords_and_zoom_to_quadKey(tile.x, tile.y, zoom)
    return quadKey

def lat_long_zoom_to_URL(x, y, zoom):
    tile = tile_address(x, y)
    quadKey = tile_coords_and_zoom_to_quadKey(tile.x, tile.y, zoom)
    tile_url = quadKey_to_url(quadKey)
    return tile_url


if __name__ == "__main__":

    zoom = 18
    outdir = '/home/ubuntu/DeepVGI-0.2/data/image_project_922_negative'
    img_dir = '/home/ubuntu/DeepVGI-0.2/data/image_project_922'
    filename = '/home/ubuntu/DeepVGI-0.2/data/all_tasks_922.csv'

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        listx, listy = [], []
        for row in reader:
            if row['project_id'] == '922':
                task_x = row['task_id'].strip().split('-')[1]
                task_y = row['task_id'].strip().split('-')[2]

                listx.append(task_x)
                listy.append(task_y)

        tilelist = zip(listx, listy)


    for i in tilelist:
        image_name = str(i[0]) + '-' + str(i[1]) + '.jpeg'
        os.chdir(img_dir)
        if not os.path.exists(image_name):
            os.chdir(outdir)
            URL = lat_long_zoom_to_URL(i[0], i[1], zoom)
            urllib.urlretrieve(URL, image_name)
            if os.path.getsize(image_name) <= 1033L:
                os.remove(image_name)
