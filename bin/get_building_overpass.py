#! /usr/bin/python

import csv
import overpy

api = overpy.Overpass()
output = '../data/building_overpy.csv'
fields = ['osm_id', 'lat', 'lon']
csvfile = open(output, 'wb')
writer = csv.writer(csvfile)
writer.writerow(fields)

result_nodes = api.query("""
    node(-16.1263042884,34.2306518555,-15.6693209848,34.8046875) [building];
    (._;>;);
    out geom;
    """)
id_node, lat_node, lon_node = [], [], []
for node in result_nodes.nodes:
    id_node.append(node.id)
    lat_node.append(node.lat)
    lon_node.append(node.lon)
print len(id_node)

result_ways = api.query("""
    way(-16.1263042884,34.2306518555,-15.6693209848,34.8046875) [building];
    (._;>;);
    out geom;
    """)
id_way, lat_way, lon_way = [], [], []
for way in result_ways.ways:
    id_way.append(way.id)
    lat_way.append('way')
    lon_way.append('way')
    for node in way.nodes:
        id_way.append(node.id)
        lat_way.append(node.lat)
        lon_way.append(node.lon)
print len(id_way)

result_relations = api.query("""
    relation(-16.1263042884,34.2306518555,-15.6693209848,34.8046875) [building];
    (._;>;);
    out geom;
    """)
id_rel, lat_rel, lon_rel = [], [], []
for rel in result_relations.relations:
    id_rel.append(rel.id)
    lat_rel.append('rel')
    lon_rel.append('rel')
print len(id_rel)

idall = id_node + id_way + id_rel
latall = lat_node + lat_way + lat_rel
lonall = lon_node + lon_way + lon_rel
for id, lat, lon in zip(idall, latall, lonall):
    row = [id, lat, lon]
    writer.writerow(row)
csvfile.close()



