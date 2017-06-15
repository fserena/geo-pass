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

__author__ = 'Fernando Serena'

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': 'cache'})
api = overpy.Overpass(url=os.environ.get('OVERPASS_API_URL', 'http://localhost:5001/api/interpreter'))


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


@cache.memoize(3600)
def is_road(way):
    tags = way['tag']
    return 'highway' in tags or 'footway' in tags


@cache.memoize(3600)
def is_building(way):
    tags = way['tag']
    return 'building' in tags and tags['building'] != 'no'


@cache.memoize(3600)
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


@cache.memoize(3600)
def query_way_lamps(id, around):
    if around:
        query = """
        way({});
        node(around:20)[highway=street_lamp][lit]["lit"!~"no"];        
        out ids;
        """.format(id)
    else:
        query = """
        way({});
        node(around:20)[highway=street_lamp][lit]["lit"!~"no"];        
        out ids;""".format(id)
    result = api.query(query)
    return filter(lambda w: w.id != int(id) and (around is None or w.id in around), result.nodes)


@cache.memoize(3600)
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


@cache.memoize(3600)
def query_building_shops(way):
    result = api.query("""way({}); out geom; >; out body;""".format(way['id']))
    poly = ["{} {}".format(result.get_node(n).lat, result.get_node(n).lon) for n in way['nd']]
    poly = ' '.join(poly)
    result = api.query("""
        node(poly:"{}") -> .in;
        (
            node.in[shop];            
        );
        out ids;
        """.format(poly))
    return result.nodes


@cache.memoize(3600)
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


@cache.memoize(3600)
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


@cache.memoize(3600)
def query_node_building(node):
    result = api.query("""
        node({});
        way(around:2)[building]["building"!~"no"];
        (._; >;);
        out geom;""".format(node['id']))
    for w in result.ways:
        poly = ["{} {}".format(n.lat, n.lon) for n in w.nodes]
        poly = ' '.join(poly)
        result = api.query("""
            node(poly:"{}") -> .in;
            node({}) -> .n;
            node.in.n;            
            out ids;
            """.format(poly, node['id']))
        if result.nodes:
            return w


@cache.memoize(3600)
def query_around(id, way=True, lat=None, lon=None, radius=None):
    type = 'way' if way else 'node'
    result = api.query("""
        {}({});
        (node(around:{},{},{}); <;) -> .all;
        (node.all[highway];
         node.all[shop];
         way.all[highway];
         way.all[footway];
         way.all[building];
        );
        out ids;        
    """.format(type, id, radius, lat, lon))
    elements = list(result.ways) + list(result.nodes)
    print len(elements)
    return map(lambda x: x.id, elements)


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
                if is_building(data):
                    data['nd'] = elm._node_ids
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


@transform
def g_node(id):
    result = api.query("""
            node({});            
            out meta;
        """.format(id))
    return list(result.nodes)


@app.route('/way/<id>')
@cache.cached(timeout=3600, key_prefix=make_cache_key)
def get_way(id):
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    radius = request.args.get('radius')
    tags = request.args.get('tags')
    tags = tags is not None

    way = g_way(id)
    if not tags:
        around = query_around(id, lat=lat, lon=lng, radius=radius) if radius else None
        if is_road(way):
            way['buildings'] = map(lambda x: 'way/{}'.format(x.id),
                                   query_way_buildings(id, around))
            way['intersect'] = map(lambda x: 'way/{}'.format(x.id),
                                   query_intersect_ways(id, around))
            way['lamps'] = map(lambda x: 'node/{}'.format(x.id),
                               query_way_lamps(id, around))
        elif is_building(way):
            way['shops'] = map(lambda x: 'node/{}'.format(x.id), query_building_shops(way))
            way['ways'] = map(lambda x: 'way/{}'.format(x.id), query_building_ways(id, around))
            surr = []
            way['surrounding'] = surr
            for w in query_surrounding_buildings(id, around):
                if w.id == int(id):
                    way['center'] = {'lat': float(w.center_lat), 'lon': float(w.center_lon)}
                else:
                    surr.append('way/{}'.format(w.id))

    response = jsonify(way)
    response.headers['Cache-Control'] = 'max-age=3600'
    return response


def _process_rel_member(m):
    if m['type'] == 'node':
        mid = 'node/{}'.format(m['ref'])
    else:
        mid = 'way/{}'.format(m['ref'])
    return {'id': mid, 'role': m['role']}


@app.route('/node/<id>')
@cache.cached(timeout=3600, key_prefix=make_tags_cache_key)
def get_node(id):
    node = g_node(id)
    tags = request.args.get('tags')
    tags = tags is not None

    if not tags:
        n_building = query_node_building(node)
        if n_building:
            node['building'] = 'way/{}'.format(n_building.id)
    response = jsonify(node)
    response.headers['Cache-Control'] = 'max-age=3600'
    return response


@app.route('/area/<id>')
@cache.cached(timeout=3600, key_prefix=make_cache_key)
def get_area(id):
    result = api.query("""area({}); out meta;""".format(id))
    area = result.areas.pop()

    response = jsonify(area.tags)
    response.headers['Cache-Control'] = 'max-age=3600'
    return response


@app.route('/elements')
@cache.cached(timeout=3600, key_prefix=make_cache_key)
def get_geo_elements():
    lat = float(request.args.get('lat'))
    lng = float(request.args.get('lng'))
    radius = int(request.args.get('radius', 200))
    limit = request.args.get('limit', None)
    if limit:
        limit = int(limit)

    poi = LatLon(lat, lng)

    # osm_result = api.query("""
    #     way(around:{},{},{}) -> .wa;
    #     node(around:{},{},{}) -> .na;
    #     (
    #         way.wa[highway]; >;
    #         way.wa[footway]; >;
    #         way.wa[building]["building"!~"no"]; >;
    #         node.na[shop];
    #         node.na[highway=street_lamp];
    #     );
    #     out geom;
    #     """.format(radius, lat, lng, radius, lat, lng))

    osm_result = api.query("""
        node(around:{},{},{}) -> .na;
        way(around:{},{},{}) -> .wa;
        (
            way.wa[highway];
            way.wa[footway];
            way.wa[building]["building"!~"no"];
            node.na[shop];
            node.na[highway=street_lamp];
        );        
        out;
        """.format(radius, lat, lng, radius, lat, lng))

    buildings = []
    ways = []
    shops = []
    lamps = []

    for i, node in enumerate(
            filter(lambda x: 'shop' in x.tags or x.tags.get('highway', '') == 'street_lamp', osm_result.nodes)):
        elm = {'id': 'node/{}'.format(node.id), 'distance': node_distance(node, poi)}
        if 'shop' in node.tags:
            shops.append(elm)
        else:
            lamps.append(elm)

        if limit and i == limit - 1:
            break

    for i, way in enumerate(osm_result.ways):
        elm = {'id': 'way/{}'.format(way.id), 'distance': way_distance(osm_result, way, poi)}
        if 'building' in way.tags:
            buildings.append(elm)
        else:
            ways.append(elm)

        if limit and i == limit - 1:
            break

    municipality = None
    province = None
    country = None
    osm_result = api.query("""
          is_in({},{}) -> .a;
          area.a[wikidata][admin_level];
          out;""".format(lat, lng))
    for area in osm_result.areas:
        a = 'area/{}'.format(area.id)
        admin_level = int(area.tags['admin_level'])
        if admin_level == 4:
            province = a
        elif admin_level == 8:
            municipality = a
        elif admin_level == 2:
            country = a

    elms_dict = {'buildings': buildings, 'ways': ways, 'shops': shops, 'lamps': lamps}
    if municipality:
        elms_dict['municipality'] = municipality
    if province:
        elms_dict['province'] = province
    if country:
        elms_dict['country'] = country

    response = jsonify(elms_dict)
    response.headers['Cache-Control'] = 'max-age=3600'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, use_reloader=False, debug=False, threaded=True)
