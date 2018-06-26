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
import calendar
import json
import os
import sys
import time
from datetime import datetime
from decimal import Decimal
from functools import wraps
from urllib2 import urlopen, HTTPError
from wsgiref.handlers import format_date_time

import overpy
import shapely
from LatLon import LatLon
from flask import Flask, request, jsonify
from flask_caching import Cache
from overpy import exception
from overpy.exception import DataIncomplete
from redis_cache import cache_it_json, SimpleCache
from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiPolygon
from shapely.ops import nearest_points
from shapely.wkt import dumps

from geo_pass import geocoding
from geo_pass.poly import ways2poly

__author__ = 'Fernando Serena'

MAX_AGE = int(os.environ.get('MAX_AGE', 86400))
CACHE_LIMIT = int(os.environ.get('CACHE_LIMIT', 100000))
CACHE_REDIS_HOST = os.environ.get('CACHE_REDIS_HOST', '127.0.0.1')
CACHE_REDIS_DB = int(os.environ.get('CACHE_REDIS_DB', 0))
CACHE_REDIS_PORT = int(os.environ.get('CACHE_REDIS_PORT', 6379))

app = Flask(__name__)
cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_KEY_PREFIX': 'geo',
    'CACHE_REDIS_HOST': CACHE_REDIS_HOST,
    'CACHE_REDIS_DB': CACHE_REDIS_DB,
    'CACHE_REDIS_PORT': CACHE_REDIS_PORT
})


# cache = Cache(app, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': 'cache'
# })


class Overpass(overpy.Overpass):
    def __init__(self, url=None, cache=None):
        super(Overpass, self).__init__(url=url)
        self.cache = cache

    def parse_json(self, data, encoding="utf-8"):
        """
        Parse raw response from Overpass service.

        :param data: Raw JSON Data
        :type data: String or Bytes
        :param encoding: Encoding to decode byte string
        :type encoding: String
        :return: Result object
        :rtype: overpy.Result
        """

        try:
            members = data.get('elements', [{}])[0].get('members', [])
        except IndexError:
            members = None
        if members:
            members = filter(lambda m: m.get('geometry', True), members)
            data['elements'][0]['members'] = members
            for m in members:
                if 'geometry' in m:
                    geometry = filter(lambda p: p, m['geometry'])
                    m['geometry'] = geometry

        return overpy.Result.from_json(data, api=self)

    def __request(self, query):
        try:
            f = urlopen(self.url, query)
        except HTTPError as e:
            f = e

        response = f.read(self.read_chunk_size)
        while True:
            data = f.read(self.read_chunk_size)
            if len(data) == 0:
                break
            response = response + data
        f.close()

        if f.code == 200:
            if overpy.PY2:
                http_info = f.info()
                content_type = http_info.getheader("content-type")
            else:
                content_type = f.getheader("Content-Type")

            if content_type == "application/json":
                if isinstance(response, bytes):
                    response = response.decode("utf-8")
                response = json.loads(response)
                return response

            raise exception.OverpassUnknownContentType(content_type)

        if f.code == 400:
            msgs = []
            for msg in self._regex_extract_error_msg.finditer(response):
                tmp = self._regex_remove_tag.sub(b"", msg.group("msg"))
                try:
                    tmp = tmp.decode("utf-8")
                except UnicodeDecodeError:
                    tmp = repr(tmp)
                msgs.append(tmp)

            raise exception.OverpassBadRequest(
                query,
                msgs=msgs
            )

        if f.code == 429:
            raise exception.OverpassTooManyRequests

        if f.code == 504:
            raise exception.OverpassGatewayTimeout

        raise exception.OverpassUnknownHTTPStatusCode(f.code)

    def query(self, query):
        if not isinstance(query, bytes):
            query = query.encode("utf-8")

        print "querying:", query
        query = "[out:json];\n" + query

        response = cache_it_json(cache=self.cache)(self.__request)(query)
        response_str = json.dumps(response)
        try:
            return self.parse_json(json.loads(response_str, parse_float=Decimal))
        except AttributeError:
            return None


cache_json = SimpleCache(hashkeys=True, host=CACHE_REDIS_HOST, port=CACHE_REDIS_PORT,
                         db=CACHE_REDIS_DB, namespace='ops', limit=100000, expire=MAX_AGE)

api = Overpass(url=os.environ.get('OVERPASS_API_URL', 'http://127.0.0.1:5000/api/interpreter'),
               cache=cache_json)


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
    return path.encode('utf-8')


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


@cache_it_json(cache=cache_json)
def is_road(way):
    tags = way.get('tag', {})
    return 'highway' in tags or 'footway' in tags


@cache_it_json(cache=cache_json)
def is_building(way):
    tags = way.get('tag', {})
    return 'amenity' in tags or ('building' in tags and tags['building'] != 'no')


@cache_it_json(cache=cache_json)
def query_way_buildings(id):
    query = """
     way({});
    way(around:20)[building]["building"!~"no"];        
    out center;""".format(id)
    result = api.query(query)
    all_way_buildings = filter(lambda w: w.id != int(id), result.ways)
    return map(lambda x: {'id': x.id, 'center_lat': float(x.center_lat), 'center_lon': float(x.center_lon)},
               all_way_buildings)


@cache_it_json(cache=cache_json)
def query_way_elms(id):
    query = """
    way({});
    (
     node(around:20)[highway];
     node(around:20)[footway];
    );        
    out body;""".format(id)
    result = api.query(query)
    return map(lambda x: {'id': x.id, 'lat': float(x.lat), 'lon': float(x.lon), 'tags': x.tags}, result.nodes)


@cache_it_json(cache=cache_json)
def query_building_ways(id):
    def filter_way(w):
        buildings = query_way_buildings(w.id)
        return any([b for b in buildings if b['id'] == int(id)])

    result = api.query("""
        way({});
        way(around:20) -> .aw;                
        (
          way.aw[highway];
          way.aw[footway];
        );
        out ids;
        """.format(id))
    all_near_ways = result.ways
    building_ways = filter(lambda w: filter_way(w), all_near_ways)
    return map(lambda x: {'id': x.id}, filter(lambda w: w.id != int(id), building_ways))


@cache_it_json(cache=cache_json)
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

    elms = []

    for n in result.nodes:
        n_point = Point(float(n.lon), float(n.lat))
        if n_point.within(shape):
            elms.append({'id': n.id, 'lat': float(n.lat), 'lon': float(n.lon), 'tags': n.tags})

    return elms


@cache_it_json(cache=cache_json)
def query_surrounding_buildings(id):
    result = api.query("""
        way({});
        out center;
        > -> .wn;
        node(around:10) -> .ar;
        node.ar.wn;
        <;        
        out center;""".format(id))
    return map(lambda x: {'id': x.id, 'center_lat': float(x.center_lat), 'center_lon': float(x.center_lon)},
               filter(lambda w: w.id != int(id) and 'building' in w.tags, result.ways))


@cache_it_json(cache=cache_json)
def query_intersect_ways(id):
    result = api.query("""
    way({});
    >;
    out body;
    < -> .wall;
    (
        way.wall[highway];
        way.wall[footway];
    );    
    out geom;""".format(id))
    node_ids = set(map(lambda x: x.id, result.nodes))
    ways = filter(lambda w: w.id != int(id), result.ways)

    intersections = {}

    for w in ways:
        cuts = set(w._node_ids).intersection(node_ids)
        if cuts:
            intersections[w.id] = list(cuts)

    return intersections


@cache_it_json(cache=cache_json)
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
            return {'id': w.id}


@cache_it_json(cache=cache_json)
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


@cache_it_json(cache=cache_json)
def query_nodes(*nodes):
    q_nodes = map(lambda x: 'node({});'.format(x), nodes)
    result = api.query("""
        ({});
        out;""".format('\n'.join(q_nodes)))

    return list(result.nodes)


def transform(f):
    def transform_value(v):
        if isinstance(v, datetime):
            return calendar.timegm(v.timetuple())
        return v

    @wraps(f)
    def wrapper(*args, **kwargs):
        data = {}
        for elm in f(*args, **kwargs):
            data['id'] = elm.id
            if elm.tags:
                data['tag'] = elm.tags
            if elm.attributes:
                data.update({k: transform_value(elm.attributes[k]) for k in elm.attributes})
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


@cache_it_json(cache=cache_json)
@transform
def g_way(id):
    center = api.query("""
                way({});   
                out center;            
            """.format(id))
    return list(center.ways)


def g_coord_area(lat, lon):
    result = api.query("""
                is_in({},{});
                out;
    """.format(lat, lon))
    return list(result.area_ids)


@cache_it_json(cache=cache_json)
def g_way_geom(id):
    geom = api.query("""
                way({});
                (._; >;);
                out geom;
            """.format(id))
    return map(lambda n: (float(n.lon), float(n.lat)), list(geom.ways).pop().nodes)


def relation_multipolygon(rels):
    rel_members = [[m for m in r.members if m.geometry and m.role == 'outer'] for r in rels]
    all_members = reduce(lambda x, y: x + y, rel_members, [])
    poly, incomplete = ways2poly(all_members)

    geoms = []
    for p in poly:
        geoms.append(Polygon([(n.lon, n.lat) for n in p]))  # .simplify(0.000001))
    for l in incomplete:
        geoms.append(Polygon(LineString([(n.lon, n.lat) for n in l])))  # .simplify(0.000001))

    return MultiPolygon(geoms)


@cache_it_json(cache=cache_json)
def g_area_geom(id):
    def transform_tag_value(k, v):
        if k == 'admin_level':
            return int(v)
        return u'"{}"'.format(v)

    result = api.query("""
                area({});
                out;
            """.format(id))

    area_tags = result.areas[0].tags
    area_tags = {key: transform_tag_value(key, v) for key, v in area_tags.items() if
                 key in ['type', 'name:en', 'name', 'admin_level', 'boundary']}

    if 'name' not in area_tags:
        return []

    tag_filters = u''.join(
        [u'["{}"={}]'.format(key, value) for key, value in area_tags.items()])

    geom = api.query(u"""            
            rel{};            
            out geom;
    """.format(tag_filters))

    if geom is None:
        return []

    if geom.relations:
        area_poly = relation_multipolygon(geom.relations)
    else:
        geom = api.query(u"""
                    way{};            
                    out body;
                    >;
                    out skel qt;
            """.format(tag_filters))
        all_points = list(geom.ways).pop().nodes
        area_poly = Polygon(map(lambda n: (float(n.lon), float(n.lat)), all_points))

    if isinstance(area_poly, Polygon):
        area_poly = MultiPolygon([area_poly])

    return [map(lambda x: tuple(x[0]), zip(p.exterior.coords)) for p in area_poly]


def get_area_multipolygon(id):
    area_geoms = g_area_geom(int(id))
    area_polys = [Polygon(points) for points in area_geoms]
    return MultiPolygon(area_polys)


@cache_it_json(cache=cache_json)
@transform
def g_node(id):
    result = api.query("""
            node({});            
            out;
        """.format(id))
    return list(result.nodes)


@cache.memoize(MAX_AGE)
def g_node_position(id):
    r = api.query("""
        node({});
        out;
    """.format(id))
    node = list(r.nodes).pop()
    return node.lat, node.lon


def nodes_in_buffer(nodes, buffer):
    for n in nodes:
        n_lat, n_lon = g_node_position(n)
        p = Point(float(n_lon), float(n_lat))
        if p.within(buffer):
            return True
    return False


def filter_building(way, b, polygons):
    near_way_points = nearest_points(polygons[b['id']], way)
    b_near_way = near_way_points[0]
    buff = b_near_way.buffer(b_near_way.distance(near_way_points[1]) + 0.00001, mitre_limit=1.0)
    buff = shapely.affinity.scale(buff, 1.0, 0.75)

    n_intersect = buff.boundary.intersection(way)
    n_intersect = [n_intersect] if isinstance(n_intersect, Point) else list(n_intersect)
    n_intersect = MultiPoint(n_intersect)

    filtered = True
    if n_intersect:
        nearest = nearest_points(polygons[b['id']], n_intersect)
        mp = MultiPoint(list(n_intersect) + [nearest[0]])

        try:
            mp = Polygon(mp)
            filtered = any(
                filter(lambda (bid, p): bid != b['id'] and p.boundary.intersects(mp.boundary),
                       polygons.items()))
        except ValueError:
            pass

    return filtered


@app.route('/way/<id>')
@cache.cached(timeout=MAX_AGE, key_prefix=make_cache_key)
def get_way(id):
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    radius = request.args.get('radius')
    area = request.args.get('area')
    tags = request.args.get('tags')
    tags = tags is not None
    buffer = None

    if area:
        buffer = get_area_multipolygon(area)
    elif any([lat, lng, radius]):
        lat = float(lat)
        lng = float(lng)
        radius = float(radius)
        poi = LatLon(lat, lng)
        r_p = poi.offset(90, radius / 1000.0)
        buffer = Point(lng, lat).buffer(abs((float(r_p.lon) - float(poi.lon))), resolution=5, mitre_limit=1.0)
        buffer = shapely.affinity.scale(buffer, 1.0, 0.75)

    way = g_way(id)
    if not tags:
        if is_road(way):
            way['type'] = 'way'
            all_w_buildings = query_way_buildings(id)
            w_buildings = filter(lambda x: not buffer or Point(x['center_lon'], x['center_lat']).within(buffer),
                                 all_w_buildings)

            shape = LineString(g_way_geom(id))

            all_building_polygons = {b['id']: Polygon(g_way_geom(b['id'])) for b in w_buildings}
            w_buildings = filter(lambda b: not filter_building(shape, b, all_building_polygons), w_buildings)

            way['buildings'] = map(lambda x: 'way/{}'.format(x['id']),
                                   w_buildings)

            all_intersects = query_intersect_ways(id)
            w_intersects = filter(lambda (w, cuts): not buffer or nodes_in_buffer(cuts, buffer), all_intersects.items())
            way['intersect'] = map(lambda x: 'way/{}'.format(x[0]), w_intersects)

            for node in query_way_elms(id):
                p = Point(node['lon'], node['lat'])
                if not buffer or p.within(buffer):
                    if 'contains' not in way:
                        way['contains'] = []
                    n_key = _elm_key(node, MATCH_TAGS)
                    if n_key is None:
                        n_key = 'node'
                    way['contains'].append({
                        'id': 'node/{}'.format(node['id']),
                        'type': n_key
                    })

        elif is_building(way):
            way['type'] = 'building'
            for node in query_building_elms(way):
                p = Point(node['lon'], node['lat'])
                if not buffer or p.within(buffer):
                    if 'contains' not in way:
                        way['contains'] = []
                    n_key = _elm_key(node, MATCH_TAGS)
                    if n_key is None:
                        n_key = 'node'
                    way['contains'].append({
                        'id': 'node/{}'.format(node['id']),
                        'type': n_key
                    })

            way['ways'] = map(lambda x: 'way/{}'.format(x['id']), query_building_ways(id))
            surr = []
            way['surrounding_buildings'] = surr
            for w in query_surrounding_buildings(id):
                if w['id'] == int(id):
                    way['center'] = {'lat': w['center_lat'], 'lon': w['center_lon']}
                else:
                    p = Point(w['center_lon'], w['center_lat'])
                    if not buffer or p.within(buffer):
                        surr.append('way/{}'.format(w['id']))

            way['areas'] = map(lambda aid: 'area/{}'.format(aid),
                               g_coord_area(way['center']['lat'], way['center']['lon']))

    del way['nd']
    response = jsonify(way)
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    response.headers['Last-Modified'] = format_date_time(time.mktime(datetime.now().timetuple()))
    return response


@app.route('/way/<id>/geom')
@cache.cached(timeout=MAX_AGE)
def get_way_geom(id):
    points = g_way_geom(id)
    way = g_way(id)
    shape = Polygon(points) if is_building(way) else LineString(points)
    response = jsonify({'wkt': dumps(shape)})
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    response.headers['Last-Modified'] = format_date_time(time.mktime(datetime.now().timetuple()))
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

    n_key = _elm_key(node, MATCH_TAGS)
    node['type'] = n_key

    if not tags:
        n_building = query_node_building(node)
        if n_building:
            node['building'] = 'way/{}'.format(n_building['id'])

        node['areas'] = map(lambda aid: 'area/{}'.format(aid),
                            g_coord_area(node['lat'], node['lon']))

    response = jsonify(node)
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    response.headers['Last-Modified'] = format_date_time(time.mktime(datetime.now().timetuple()))
    return response


@app.route('/node/<id>/geom')
@cache.cached(timeout=MAX_AGE, key_prefix=make_tags_cache_key)
def get_node_geom(id):
    node = g_node(id)
    point = Point(node['lon'], node['lat'])
    response = jsonify({'wkt': dumps(point)})
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    response.headers['Last-Modified'] = format_date_time(time.mktime(datetime.now().timetuple()))
    return response


def search_sub_admin_areas(sub_admin_level, geoms):
    cur_admin_level = sub_admin_level

    all_sub_areas_result = {}

    while not all_sub_areas_result and cur_admin_level < 11:
        cur_admin_level += 1
        if cur_admin_level == 5:
            cur_admin_level += 1  # skip religious admins (level 5)

        for geom in geoms:
            geom_points = map(lambda x: tuple(x[0]), zip(geom.convex_hull.exterior.coords))
            lat_lng_points = ['{} {}'.format(p[1], p[0]) for p in geom_points]
            poly_str = 'poly:"{}"'.format(' '.join(lat_lng_points))

            sub_areas_result = api.query("""
                (
                  rel[boundary][type][admin_level={}][name]({});
                  way[boundary][type][admin_level={}][name]({});
                );
                out tags;
                out center;
            """.format(cur_admin_level, poly_str, cur_admin_level, poly_str))
            for r in sub_areas_result.relations:
                r_id = 'r{}'.format(r.id)
                if r_id.format(r.id) not in all_sub_areas_result:
                    all_sub_areas_result[r_id] = r
            for w in sub_areas_result.ways:
                w_id = 'w{}'.format(w.id)
                if w_id.format(w.id) not in all_sub_areas_result:
                    all_sub_areas_result[w_id] = w

    return all_sub_areas_result.values(), cur_admin_level


@cache_it_json(cache=cache_json)
def query_sub_areas(id, admin_level):
    contained_areas = set()
    if admin_level >= 0:
        geoms = get_area_multipolygon(id)
        sub_admins_ids = set()
        sub_admin_level = admin_level

        while not contained_areas and sub_admin_level < 11:
            sub_admins, sub_admin_level = search_sub_admin_areas(sub_admin_level, geoms)
            sub_area_ids = set()
            for a in sub_admins:
                if a.id not in sub_admins_ids:
                    sub_admins_ids.add(a.id)
                else:
                    continue
                sub_area_res = api.query(u"""
                    area[boundary][type][admin_level={}][name="{}"];
                    out ids;
                """.format(sub_admin_level, a.tags['name']))
                for sub_rel_area in sub_area_res.areas:
                    sub_area_ids.add(sub_rel_area.id)

            sub_area_ids = filter(lambda a: a not in contained_areas, sub_area_ids)
            for sub_area_id in sub_area_ids:
                sub_area_multipoly = get_area_multipolygon(sub_area_id)
                for sub_area_poly in sub_area_multipoly:
                    if sub_area_poly.area <= geoms.area and sub_area_poly.within(geoms):
                        contained_areas.add(sub_area_id)
    return list(contained_areas)


@app.route('/area/<id>')
@cache.cached(timeout=MAX_AGE, key_prefix=make_cache_key)
def get_area(id):
    result = api.query("""area({}); out;""".format(id))
    area = result.areas.pop()

    admin_level = int(area.tags.get('admin_level', -1))
    spec_type = None
    if admin_level == 4:
        spec_type = 'province'
    elif admin_level == 8:
        spec_type = 'municipality'
    elif admin_level == 2:
        spec_type = 'country'
    type = 'area'

    if spec_type:
        type = ':'.join([type, spec_type])

    contained_areas = query_sub_areas(id, admin_level)

    element = {'type': type,
               'id': id,
               'tags': area.tags,
               'contains': list(map(lambda a: 'area/{}'.format(a), contained_areas))}
    response = jsonify(element)
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    response.headers['Last-Modified'] = format_date_time(time.mktime(datetime.now().timetuple()))
    return response


@app.route('/area/<id>/geom')
@cache.cached(timeout=MAX_AGE)
def get_area_geom(id):
    multipolygon = get_area_multipolygon(id)
    n_nodes = reduce(lambda x, y: x + len(y.exterior.coords), multipolygon, 0)
    simpl_linear_factor = 0.0000006711
    simpl_factor = simpl_linear_factor * n_nodes
    if simpl_factor > 0.01:
        simpl_factor = 0.01
    multipolygon = multipolygon.simplify(simpl_factor)
    response = jsonify({'wkt': dumps(multipolygon)})
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    response.headers['Last-Modified'] = format_date_time(time.mktime(datetime.now().timetuple()))
    return response


def _elm_key(elm, match=set()):
    key = None
    tags = elm.get('tags', elm.get('tag'))
    matching_tags = list(set(match).intersection(set(tags.keys())))
    try:
        tag = matching_tags.pop()
        key = '{}:{}'.format(tag, tags[tag])
    except IndexError:
        pass

    return key


MATCH_TAGS = {'shop', 'highway', 'amenity', 'building', 'tourism'}


@app.route('/elements')
@cache.cached(timeout=MAX_AGE, key_prefix=make_cache_key)
def get_geo_elements():
    limit = request.args.get('limit', None)
    if limit:
        limit = int(limit)

    area = request.args.get('area')
    area_geoms = None
    if area:
        area_geoms = g_area_geom(area) if area else None
        mp = get_area_multipolygon(area)
        print mp.area
    else:
        try:
            location = request.args.get('location')
            ll = geocoding(location)
            lat, lng = ll['lat'], ll['lng']
        except Exception:
            try:
                lat = float(request.args.get('lat'))
                lng = float(request.args.get('lng'))
            except TypeError:
                response = jsonify({'message': 'Bad arguments'})
                response.status_code = 400
                return response

    if area_geoms:
        for points in area_geoms:
            lat, lng = points[0][1], points[0][0]
            lat_lng_points = ['{} {}'.format(p[1], p[0]) for p in points]
            geo_filter = 'poly:"{}"'.format(' '.join(lat_lng_points))
            center_tuple = list(Polygon(points).representative_point().coords)[0]
            center = LatLon(*reversed(center_tuple))
    else:
        radius = int(request.args.get('radius', 200))
        center = LatLon(lat, lng)
        geo_filter = 'around:{},{},{}'.format(radius, lat, lng)

    osm_result = api.query("""
        node({}) -> .na;
        way({}) -> .wa;
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
        """.format(geo_filter, geo_filter))

    elms = []

    for i, node in enumerate(osm_result.nodes):
        elm = {'id': 'node/{}'.format(node.id)}  # node_distance(node, poi)}
        key = _elm_key({'tags': node.tags}, MATCH_TAGS)
        if key is None:
            key = 'node'
        if 'name' in node.tags:
            elm['name'] = node.tags['name']
        elm['type'] = 'node:' + key
        elms.append(elm)

        if limit and i == limit - 1:
            break

    for i, way in enumerate(osm_result.ways):
        elm = {'id': 'way/{}'.format(way.id)}  # way_distance(osm_result, way, poi)}
        key = _elm_key({'tags': way.tags}, MATCH_TAGS)
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
            'lat': float(center.lat),
            'lng': float(center.lon)
        },
        'results': elms
    }

    response = jsonify(result)
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    response.headers['Last-Modified'] = format_date_time(time.mktime(datetime.now().timetuple()))
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, use_reloader=False, debug=False, threaded=True)
