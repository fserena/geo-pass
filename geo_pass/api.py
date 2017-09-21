"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Ontology Engineering Group
        http://www.oeg-upm.net/
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2016 Ontology Engineering Group.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
"""
import os
import sys
from functools import wraps

import overpy
from LatLon import LatLon
from flask import Flask, request, jsonify
from overpy.exception import DataIncomplete
from flask_cache import Cache
from shapely.geometry import Point, Polygon, LineString
from shapely.wkt import dumps

from geo_pass import geocoding

__author__ = 'Fernando Serena'

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': 'cache', 'CACHE_THRESHOLD': 100000})
api = overpy.Overpass(url=os.environ.get('OVERPASS_API_URL', 'http://localhost:5001/api/interpreter'))

MAX_AGE = int(os.environ.get('MAX_AGE', 86400))


def make_cache_key(*args, **kwargs):
    path = request.path
    qargs = dict(request.args.items())
    args = ''.join(['{}{}'.format(k, qargs[k]) for k in sorted(qargs.keys())])
    return (path + args).encode('utf-8')


def make_around_cache_key(*args, **kwargs):
    qargs = dict(request.args.items())
    args = 'around' + ''.join(['{}{}'.format(k, qargs[k]) for k in sorted(qargs.keys())])
    return args.encode('utf-8')


def make_tags_cache_key(*args, **kwargs):
    path = request.path
    qargs = dict(request.args.items())
    args = 'tags' + ''.join(['{}{}'.format(k, qargs[k]) for k in sorted(qargs.keys())])
    return (path + args).encode('utf-8')


def way_distance(result, way, point):
    min = sys.maxint
    for nid in way._node_ids:
        try:
            node = result.get_node(nid, resolve_missing=False)
            node_ll = LatLon(node.lat, node.lon)
            node_d = point.distance(node_ll)
            if node_d < min:
                min = node_d
        except DataIncomplete:
            pass
    return min


def node_distance(node, point):
    node_ll = LatLon(node.lat, node.lon)
    return point.distance(node_ll)


@cache.memoize(MAX_AGE)
def is_road(way):
    tags = way['tag']
    return 'highway' in tags or 'footway' in tags


@cache.memoize(MAX_AGE)
def is_building(way):
    tags = way['tag']
    return 'amenity' in tags or ('building' in tags and tags['building'] != 'no')


@cache.memoize(MAX_AGE)
def query_way_buildings(id, around):
    if around:
        query = """
        way({});
        way(around:20)[building]["building"!~"no"];                
        out ids;
        """.format(id)
    else:
        query = """
         way({});
        way(around:20)[building]["building"!~"no"];        
        out ids;""".format(id)
    result = api.query(query)
    return filter(lambda w: w.id != int(id) and (around is None or w.id in around), result.ways)


@cache.memoize(MAX_AGE)
def query_way_elms(id, around):
    if around:
        query = """
        way({});
        (
         node(around:20)[highway];
         node(around:20)[footway];
        );
        out;
        """.format(id)
    else:
        query = """
        way({});
        (
         node(around:20)[highway];
         node(around:20)[footway];
        );        
        out;""".format(id)
    result = api.query(query)
    return filter(lambda w: w.id != int(id) and (around is None or w.id in around), result.nodes)


@cache.memoize(MAX_AGE)
def query_building_ways(id, around):
    result = api.query("""
        way({});
        way(around:20) -> .aw;                
        (
          way.aw[highway];
          way.aw[footway];
        );
        out ids;
        """.format(id, around))
    return filter(lambda w: w.id != int(id) and (around is None or w.id in around), result.ways)


@cache.memoize(MAX_AGE)
def query_building_elms(way):
    result = api.query("""way({}); out geom; >; out body;""".format(way['id']))

    points = [(float(result.get_node(n).lon), float(result.get_node(n).lat)) for n in way['nd']]
    shape = Polygon(points).envelope

    result = api.query("""
        way({});
        node(around:5) -> .na;
        (
           node.na[shop];
           node.na[amenity];
           node.na[tourism];
        );                
        out;
        """.format(way['id']))

    for n in result.nodes:
        n_point = Point(float(n.lon), float(n.lat))
        if n_point.within(shape):
            yield n


@cache.memoize(MAX_AGE)
def query_surrounding_buildings(id, around):
    result = api.query("""
        way({});
        out center;
        > -> .wn;
        node(around:10) -> .ar;
        node.ar.wn;
        <;        
        out ids;""".format(id, around))
    return filter(lambda w: w.id != int(id) and (around is None or w.id in around), result.ways)


@cache.memoize(MAX_AGE)
def query_intersect_ways(id, around, lat=None, lon=None, radius=None):
    if radius:
        f = '> -> .wn; node.wn(around:{},{},{})'.format(radius, lat, lon)
    else:
        f = '>'
    result = api.query("""
    way({});
    {};
    < -> .wall;
    (
        way.wall[highway];
        way.wall[footway];
    );    
    out ids;""".format(id, f))
    return filter(lambda w: w.id != int(id) and (around is None or w.id in around), result.ways)


@cache.memoize(MAX_AGE)
def query_node_building(node):
    result = api.query("""
        node({});
        way(around:10)[building]["building"!~"no"];
        (._; >;);
        out geom;""".format(node['id']))
    point = Point(node['lon'], node['lat'])
    for w in result.ways:
        geom = w.nodes
        points = map(lambda n: [float(n.lon), float(n.lat)], geom)
        shape = Polygon(points)
        if point.within(shape.envelope):
            return w


@cache.memoize(MAX_AGE)
def query_around(id, way=True, lat=None, lon=None, radius=None):
    type = 'way' if way else 'node'
    result = api.query("""
        {}({});
        (node(around:{},{},{}); <;) -> .all;
        (node.all[highway];
         node.all[shop];
         node.all[amenity];
         node.all[tourism];
         way.all[highway];
         way.all[footway];
         way.all[building];
        );
        out ids;        
    """.format(type, id, radius, lat, lon))
    elements = list(result.ways) + list(result.nodes)
    return map(lambda x: x.id, elements)


@cache.memoize(MAX_AGE)
def query_nodes(*nodes):
    q_nodes = map(lambda x: 'node({});'.format(x), nodes)
    result = api.query("""
        ({});
        out;""".format('\n'.join(q_nodes)))

    return list(result.nodes)


def transform(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        data = {}
        for elm in f(*args, **kwargs):
            data['id'] = elm.id
            if elm.tags:
                data['tag'] = elm.tags
            if elm.attributes:
                data.update(elm.attributes)
            if isinstance(elm, overpy.Way):
                data['nd'] = elm._node_ids
                if is_building(data):
                    if elm.center_lat:
                        data['center'] = {'lat': float(elm.center_lat), 'lon': float(elm.center_lon)}
            else:
                data['lat'] = float(elm.lat)
                data['lon'] = float(elm.lon)
        return data

    return wrapper


@transform
def g_way(id):
    center = api.query("""
                way({});   
                out center;            
            """.format(id))
    return list(center.ways)


def g_way_geom(id):
    geom = api.query("""
                way({});
                (._; >;);
                out geom;
            """.format(id))
    return list(geom.ways).pop().nodes


@transform
def g_node(id):
    result = api.query("""
            node({});            
            out meta;
        """.format(id))
    return list(result.nodes)


@app.route('/way/<id>')
@cache.cached(timeout=MAX_AGE, key_prefix=make_cache_key)
def get_way(id):
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    radius = request.args.get('radius')
    tags = request.args.get('tags')
    tags = tags is not None

    if any([lat, lng, radius]):
        lat = float(lat)
        lng = float(lng)
        radius = float(radius)

    way = g_way(id)
    if not tags:
        around = query_around(id, lat=lat, lon=lng, radius=radius) if radius else None
        if is_road(way):
            way['buildings'] = map(lambda x: 'way/{}'.format(x.id),
                                   query_way_buildings(id, around))
            way['intersect'] = map(lambda x: 'way/{}'.format(x.id),
                                   query_intersect_ways(id, around))

            for node in query_way_elms(id, around):
                if 'contains' not in way:
                    way['contains'] = []
                n_key = _elm_key(node, MATCH_TAGS)
                if n_key is None:
                    n_key = 'node'
                way['contains'].append({
                    'id': 'node/{}'.format(node.id),
                    'type': n_key
                })

        elif is_building(way):
            for node in query_building_elms(way):
                if 'contains' not in way:
                    way['contains'] = []
                n_key = _elm_key(node, MATCH_TAGS)
                if n_key is None:
                    n_key = 'node'
                way['contains'].append({
                    'id': 'node/{}'.format(node.id),
                    'type': n_key
                })

            way['ways'] = map(lambda x: 'way/{}'.format(x.id), query_building_ways(id, around))
            surr = []
            way['surrounding_buildings'] = surr
            for w in query_surrounding_buildings(id, around):
                if w.id == int(id):
                    way['center'] = {'lat': float(w.center_lat), 'lon': float(w.center_lon)}
                else:
                    surr.append('way/{}'.format(w.id))

    del way['nd']
    response = jsonify(way)
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    return response


@app.route('/way/<id>/geom')
@cache.cached(timeout=MAX_AGE, key_prefix=make_cache_key)
def get_way_geom(id):
    geom = g_way_geom(id)
    points = map(lambda n: (float(n.lon), float(n.lat)), geom)

    way = g_way(id)
    shape = Polygon(points) if is_building(way) else LineString(points)
    response = jsonify({'wkt': dumps(shape)})
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    return response


def _process_rel_member(m):
    if m['type'] == 'node':
        mid = 'node/{}'.format(m['ref'])
    else:
        mid = 'way/{}'.format(m['ref'])
    return {'id': mid, 'role': m['role']}


@app.route('/node/<id>')
@cache.cached(timeout=MAX_AGE, key_prefix=make_tags_cache_key)
def get_node(id):
    node = g_node(id)
    tags = request.args.get('tags')
    tags = tags is not None

    if not tags:
        n_building = query_node_building(node)
        if n_building:
            node['building'] = 'way/{}'.format(n_building.id)
    response = jsonify(node)
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    return response


@app.route('/node/<id>/geom')
@cache.cached(timeout=MAX_AGE, key_prefix=make_tags_cache_key)
def get_node_geom(id):
    node = g_node(id)
    point = Point(node['lon'], node['lat'])
    response = jsonify({'wkt': dumps(point)})
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    return response


@app.route('/area/<id>')
@cache.cached(timeout=MAX_AGE, key_prefix=make_cache_key)
def get_area(id):
    result = api.query("""area({}); out meta;""".format(id))
    area = result.areas.pop()

    response = jsonify(area.tags)
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    return response


def _elm_key(elm, match=set()):
    key = None
    matching_tags = list(set(match).intersection(set(elm.tags.keys())))
    try:
        tag = matching_tags.pop()
        key = '{}:{}'.format(tag, elm.tags[tag])
    except IndexError:
        pass

    return key


MATCH_TAGS = {'shop', 'highway', 'amenity', 'building', 'tourism'}


@app.route('/elements')
@cache.cached(timeout=MAX_AGE, key_prefix=make_cache_key)
def get_geo_elements():
    location = request.args.get('location')
    if location:
        ll = geocoding(location)
        lat, lng = ll['lat'], ll['lng']
    else:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
    radius = int(request.args.get('radius', 200))
    limit = request.args.get('limit', None)
    if limit:
        limit = int(limit)

    poi = LatLon(lat, lng)

    osm_result = api.query("""
        node(around:{},{},{}) -> .na;
        way(around:{},{},{}) -> .wa;
        (
            way.wa[highway];
            way.wa[footway];
            way.wa[building]["building"!~"no"];
            node.na[shop];
            node.na[amenity];
            node.na[highway];
            node.na[tourism];
        );        
        out geom;
        """.format(radius, lat, lng, radius, lat, lng))

    elms = []

    for i, node in enumerate(osm_result.nodes):
        elm = {'id': 'node/{}'.format(node.id), 'distance': node_distance(node, poi)}
        key = _elm_key(node, MATCH_TAGS)
        if key is None:
            key = 'node'
        if 'name' in node.tags:
            elm['name'] = node.tags['name']
        elm['type'] = 'node:' + key
        elms.append(elm)

        if limit and i == limit - 1:
            break

    for i, way in enumerate(osm_result.ways):
        elm = {'id': 'way/{}'.format(way.id), 'distance': way_distance(osm_result, way, poi)}
        key = _elm_key(way, MATCH_TAGS)
        if key is None:
            key = 'way'
        if 'name' in way.tags:
            elm['name'] = way.tags['name']
        elm['type'] = 'way:' + key
        elms.append(elm)

        if limit and i == limit - 1:
            break

    osm_result = api.query("""
          is_in({},{}) -> .a;
          area.a[admin_level];
          out;""".format(lat, lng))
    for area in osm_result.areas:
        a = 'area/{}'.format(area.id)
        admin_level = int(area.tags['admin_level'])
        type = 'area'
        if admin_level == 4:
            type = 'province'
        elif admin_level == 8:
            type = 'municipality'
        elif admin_level == 2:
            type = 'country'
        elm = {'id': a, 'type': type}
        elms.append(elm)

    result = {
        'center': {
            'lat': float(poi.lat),
            'lng': float(poi.lon)
        },
        'results': elms
    }

    response = jsonify(result)
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, use_reloader=False, debug=False, threaded=True)
