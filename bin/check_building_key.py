#! /usr/bin/python

import overpy
import sys
sys.path.append("../lib")
import FileIO

api = overpy.Overpass()

new_building = '../data/building_overpy.csv'
buildings = FileIO.read_lines(new_building, 0)
building_id = []
for building in buildings:
    item = building.strip().split(',')
    osm_id = item[0]
    building_id.append(osm_id)

lines = FileIO.read_lines('../data/buildings.csv', 0)
not_know = 0
for line in lines:
    item = line.strip().split(',')
    osm_id = item[0]
    if osm_id not in building_id:
        try:
            result = api.query("way({}); out meta;".format(int(osm_id)))
            way = result.get_way(int(osm_id))
            tags = way.tags
        except:
            try:
                result = api.query("node({}); out meta;".format(int(osm_id)))
                node = result.get_node(int(osm_id))
                tags = node.tags
            except:
                try:
                    result = api.query("rel({}); out meta;".format(int(osm_id)))
                    rel = result.get_relation(int(osm_id))
                    tags = rel.tags
                except:
                    print '%s is not osm element' % osm_id
                else:
                    if 'building' not in tags:
                        not_know += 1
                        print '%s has no building key' % osm_id

print not_know
