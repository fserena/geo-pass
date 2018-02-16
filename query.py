# coding=utf-8
import geojson
import requests
from concurrent.futures import ThreadPoolExecutor
from py2neo import Graph, Path
from shapely.wkt import loads

# from crawler import Crawler

g = Graph(host='127.0.0.1', password='neo219')

pool = ThreadPoolExecutor(max_workers=2)


# def query(location, cypher):
#     crawler = Crawler(location=location, pool=pool)


if __name__ == '__main__':

    features = []

    data = g.data('MATCH (w:Way {name: "Calle Cedros"})-[:GOES_BY]->(b) RETURN DISTINCT w, b')
    # data = g.data('MATCH (w:Way {name: "Plaza del Este"})-[:GOES_BY]->(b) RETURN DISTINCT w, b')

    # data = g.data('MATCH (s:Amenity),(r:Amenity), p = shortestPath((s)-[*]-(r)) WHERE (s.latitude>40.4722280 AND s.latitude <40.4722300 AND s <> r) RETURN p')
    # data = g.data('MATCH (s:Pharmacy),(r:Confectionery), p = shortestPath((s)-[*]-(r)) RETURN p')
    #
    # data = g.data(
    #     'MATCH (n:Amenity)<-[:CONTAINS]-(b)<-[:GOES_BY]-(w)-[:GOES_BY]->(b2)-[:CONTAINS]->(s:Shop) RETURN n, w, b, b2, s')

    for r in data:
        for rsub in r.values():
            if isinstance(rsub, Path):
                nodes = rsub.nodes()
            else:
                nodes = [rsub]
            for node in nodes:
                if node:
                    node_osm_id = node['osm_id']
                    response = requests.get('http://127.0.0.1:5006/{}/geom'.format(node['api_path']))
                    if response.status_code == 200:
                        wkt = response.json()
                        elm = loads(wkt['wkt'])
                        features.append(geojson.Feature(geometry=elm, properties=dict(node)))

    collection = geojson.FeatureCollection(features)
    print collection


