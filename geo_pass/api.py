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
import ssl
import sys
import time
from datetime import datetime
from decimal import Decimal
from functools import wraps
from urllib2 import urlopen, HTTPError, Request
from wsgiref.handlers import format_date_time

import overpy
import shapely
from LatLon import LatLon
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from flask_caching import Cache
from fuzzywuzzy import process
from overpy import exception
from overpy.exception import DataIncomplete
from redis_cache import cache_it_json, SimpleCache
from shapely.errors import TopologicalError
from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiPolygon
from shapely.ops import nearest_points
from shapely.wkt import dumps

from geo_pass import geocoding, debug, error, ZSimpleCache
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


class Overpass(overpy.Overpass):
    def __init__(self, url=None, cache=None, jwt=None):
        super(Overpass, self).__init__(url=url)
        self.cache = cache
        self.jwt = jwt
        self.pool = ThreadPoolExecutor(max_workers=4)

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
            elements = data.get('elements', [{}])[0]
            if elements.get('type', '') == 'count':
                return {k: int(v) for k, v in elements['tags'].iteritems()}
            else:
                members = elements.get('members', [])

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
        debug("querying:" + query)

        if self.url.startswith('https'):
            gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)
        else:
            gcontext = None

        if self.jwt:
            request = Request(self.url, headers={"Authorization": self.jwt})
        else:
            request = self.url

        future = self.pool.submit(urlopen, request, query, context=gcontext)

        try:
            f = future.result()
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

    def query(self, query, cache=True, expire=MAX_AGE, maxsize=536870912):
        if not isinstance(query, bytes):
            query = query.encode("utf-8")

        query = "[maxsize:{}][out:json];\n".format(maxsize) + query

        if cache:
            opt_query = ';'.join([st.strip('\n').strip(' ') for st in query.split(';')])
            try:
                response = cache_it_json(cache=self.cache, expire=expire)(self.__request)(opt_query)
            except Exception as e:
                raise ValueError(e.message)
        else:
            response = self.__request(query)

        response_str = json.dumps(response)
        try:
            return self.parse_json(json.loads(response_str, parse_float=Decimal))
        except AttributeError:
            return None


cache_proc = SimpleCache(hashkeys=True, host=CACHE_REDIS_HOST, port=CACHE_REDIS_PORT,
                         db=CACHE_REDIS_DB + 1, namespace='pr', limit=10000000, expire=MAX_AGE)

cache_q = ZSimpleCache(hashkeys=True, host=CACHE_REDIS_HOST, port=CACHE_REDIS_PORT,
                       db=CACHE_REDIS_DB + 2, namespace='q', limit=100000, expire=MAX_AGE)

api = Overpass(url=os.environ.get('OVERPASS_API_URL', 'http://127.0.0.1:5000/api/interpreter'),
               cache=cache_q, jwt=os.environ.get('JWT_TOKEN', None))


def make_cache_key(*args, **kwargs):
    path = request.path
    qargs = dict(request.args.items())
    args = 'u'.join([u'{}{}'.format(k, qargs[k]) for k in sorted(qargs.keys())])
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


def is_road(way):
    tags = way.get('tag', {})
    return 'highway' in tags or 'footway' in tags


def is_building(way):
    tags = way.get('tag', {})
    return 'amenity' in tags or ('building' in tags and tags['building'] != 'no')


@cache_it_json(cache=cache_proc)
def query_way_buildings(id):
    debug('Way buildings of ' + str(id))
    query = """
     way({});
    way(around:20)[building]["building"!~"no"];        
    out center;""".format(id)
    result = api.query(query, cache=False)
    all_way_buildings = filter(lambda w: w.id != int(id), result.ways)
    return map(lambda x: {'id': x.id, 'center_lat': float(x.center_lat), 'center_lon': float(x.center_lon)},
               all_way_buildings)


@cache_it_json(cache=cache_proc)
def query_way_elms(id):
    debug('Way elements of ' + str(id))
    query = """
    way({});
    (
     node(around:20)[highway];
     node(around:20)[footway];
    );        
    out body;""".format(id)
    result = api.query(query, cache=False)
    return map(lambda x: {'id': x.id, 'lat': float(x.lat), 'lon': float(x.lon), 'tags': x.tags}, result.nodes)


@cache_it_json(cache=cache_proc)
def query_building_ways(id):
    debug('Building ways of ' + str(id))

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
        """.format(id), cache=False)
    all_near_ways = result.ways
    building_ways = filter(lambda w: filter_way(w), all_near_ways)
    return map(lambda x: {'id': x.id}, filter(lambda w: w.id != int(id), building_ways))


@cache_it_json(cache=cache_proc)
def query_building_elms(way):
    debug('Building elements of ' + str(way['id']))
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
        """.format(way['id']), cache=False)

    elms = []

    for n in result.nodes:
        n_point = Point(float(n.lon), float(n.lat))
        if n_point.within(shape):
            elms.append({'id': n.id, 'lat': float(n.lat), 'lon': float(n.lon), 'tags': n.tags})

    return elms


@cache_it_json(cache=cache_proc)
def query_surrounding_buildings(id):
    debug('Surrounding buildings of ' + str(id))
    result = api.query("""
        way({});
        out center;
        > -> .wn;
        node(around:10) -> .ar;
        node.ar.wn;
        <;        
        out center;""".format(id), cache=False)
    return map(lambda x: {'id': x.id, 'center_lat': float(x.center_lat), 'center_lon': float(x.center_lon)},
               filter(lambda w: w.id != int(id) and 'building' in w.tags, result.ways))


@cache_it_json(cache=cache_proc)
def query_intersect_ways(id):
    debug('Intersecting ways of ' + str(id))
    result = api.query("""
    way({});
    >;
    out body;
    < -> .wall;
    (
        way.wall[highway];
        way.wall[footway];
    );    
    out geom;""".format(id), cache=False)
    node_ids = set(map(lambda x: x.id, result.nodes))
    ways = filter(lambda w: w.id != int(id), result.ways)

    intersections = {}

    for w in ways:
        cuts = set(w._node_ids).intersection(node_ids)
        if cuts:
            intersections[w.id] = list(cuts)

    return intersections


@cache_it_json(cache=cache_proc)
def query_node_building(node):
    result = api.query("""
        node({});
        way(around:10)[building]["building"!~"no"];
        (._; >;);
        out geom;""".format(node['id']), cache=False)
    point = Point(node['lon'], node['lat'])
    for w in result.ways:
        geom = w.nodes
        points = map(lambda n: [float(n.lon), float(n.lat)], geom)
        shape = Polygon(points)
        if point.within(shape.envelope):
            return {'id': w.id}


@cache_it_json(cache=cache_proc)
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
    """.format(type, id, radius, lat, lon), cache=False)
    elements = list(result.ways) + list(result.nodes)
    return map(lambda x: x.id, elements)


@cache_it_json(cache=cache_proc)
def query_nodes(*nodes):
    q_nodes = map(lambda x: 'node({});'.format(x), nodes)
    result = api.query("""
        ({});
        out;""".format('\n'.join(q_nodes)), cache=False)

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


@cache_it_json(cache=cache_proc)
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
                out tags;
    """.format(lat, lon))

    max_admin_level = max([int(a.tags['admin_level']) for a in result.areas if 'admin_level' in a.tags])
    return [a.id for a in result.areas if 'admin_level' in a.tags and
            a.tags.get('boundary', '') == 'administrative' and int(a.tags['admin_level']) == max_admin_level]


@cache_it_json(cache=cache_proc)
def g_way_geom(id):
    debug('Geometry of ' + str(id))
    geom = api.query("""
                way({});
                (._; >;);
                out geom;
            """.format(id), cache=False)
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


@cache_it_json(cache=cache_proc)
def g_area_geom(id):
    debug('Area geometry of ' + str(id))

    def transform_tag_value(k, v):
        if k == 'admin_level':
            return int(v)
        return u'"{}"'.format(v)

    result = api.query("""area({});out;""".format(id))

    area_tags = result.areas[0].tags
    area_tags = {key: transform_tag_value(key, v) for key, v in area_tags.items() if
                 key in ['type', 'name:en', 'name', 'boundary', 'wikidata', 'is_in']}

    if 'name' not in area_tags:
        return []

    geom = api.query(u"""    
            area({});        
            rel(pivot);            
            out geom;
    """.format(id), cache=False)

    if geom is None:
        return []

    tag_filters = u''.join(
        [u'["{}"={}]'.format(key, value) for key, value in area_tags.items()])

    if not geom.relations:
        geom = api.query(u"""    
                    rel{};            
                    out geom;
            """.format(tag_filters), cache=False)

    if geom.relations:
        area_poly = relation_multipolygon(geom.relations)
    else:
        geom = api.query(u"""
                    area({});
                    way(pivot);            
                    out geom;
                    >;
                    out skel qt;
            """.format(id), cache=False)
        all_points = list(geom.ways).pop().nodes
        area_poly = Polygon(map(lambda n: (float(n.lon), float(n.lat)), all_points))

    if isinstance(area_poly, Polygon):
        area_poly = MultiPolygon([area_poly])

    return [map(lambda x: tuple(x[0]), zip(p.exterior.coords)) for p in area_poly]


def get_area_multipolygon(id):
    area_geoms = g_area_geom(str(id))
    area_polys = [Polygon(points) for points in area_geoms]
    return MultiPolygon(area_polys)


@cache_it_json(cache=cache_proc)
@transform
def g_node(id):
    result = api.query("""
            node({});            
            out;
        """.format(id))
    return list(result.nodes)


@cache_it_json(cache=cache_proc)
def g_node_position(id):
    r = api.query("""
        node({});
        out;
    """.format(id))
    node = list(r.nodes).pop()
    return float(node.lat), float(node.lon)


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


def get_way_attrs(id, tags, buffer):
    way = g_way(id)
    if is_road(way):
        way['type'] = 'way'
        if not tags:
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
        if not tags:
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

    if 'nd' in way:
        del way['nd']
    return way


@app.route('/way/<id>')
@cache.cached(timeout=MAX_AGE, key_prefix=make_cache_key)
def get_way(id):
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    area = request.args.get('area')
    tags = request.args.get('tags')
    tags = tags is not None
    buffer = None

    if area:
        try:
            int(area)
        except ValueError:
            area = match_area_id(area)

        buffer = get_area_multipolygon(area)
    elif any([lat, lng]):
        lat = float(lat)
        lng = float(lng)
        radius = float(request.args.get('radius', 100))
        poi = LatLon(lat, lng)
        r_p = poi.offset(90, radius / 1000.0)
        buffer = Point(lng, lat).buffer(abs((float(r_p.lon) - float(poi.lon))), resolution=5, mitre_limit=1.0)
        buffer = shapely.affinity.scale(buffer, 1.0, 0.75)

    way = get_way_attrs(id, tags, buffer)
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
    debug('Getting node ' + str(id))
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

        if not isinstance(geoms, MultiPolygon):
            geoms = MultiPolygon([geoms])

        for geom in geoms:
            geom_points = map(lambda x: tuple(x[0]), zip(geom.convex_hull.exterior.coords))
            lat_lng_points = ['{} {}'.format(p[1], p[0]) for p in geom_points]
            poly_str = 'poly:"{}"'.format(' '.join(lat_lng_points))

            sub_areas_result = api.query("""
                (
                  rel[boundary=administrative][type][admin_level={}][name]({});
                  way[boundary=administrative][type][admin_level={}][name]({});
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


@cache_it_json(cache=cache_proc)
def area_sfc(id):
    sub_area_multipoly = get_area_multipolygon(id)
    return sub_area_multipoly.area


@cache_it_json(cache=cache_proc)
def query_sub_areas(id, admin_level):
    def next_admin_level():
        if admin_level <= 2:
            return sub_admin_level < 4
        elif admin_level < 8:
            return sub_admin_level - admin_level <= 3
        elif sub_admin_level < 11:
            return sub_admin_level - admin_level <= 2

        return False

    area_names = g_area_names()

    contained_areas = set()
    admin_level = int(admin_level)
    if admin_level >= 0:
        geoms = get_area_multipolygon(id)
        diff_geoms = get_area_multipolygon(id)
        area_factor = simplify_factor(geoms, max=0.1)
        simpl_geoms = geoms.simplify(area_factor)
        sub_admins_ids = set()
        sub_admin_level = admin_level
        geoms_area = geoms.area

        while next_admin_level() and (diff_geoms.area > geoms_area * 0.1 or not contained_areas):
            bounds = diff_geoms.bounds

            if len(bounds) != 4:
                break

            sub_admins, sub_admin_level = search_sub_admin_areas(sub_admin_level, diff_geoms)
            sub_area_ids = set()
            for a in sub_admins:
                if a.id not in sub_admins_ids:
                    sub_admins_ids.add(a.id)
                else:
                    continue

                matching_sub_ids = filter(lambda an: an['l'] == sub_admin_level, area_names.get(a.tags['name'], {}))
                if len(matching_sub_ids) == 1:
                    sub_area_ids.add(matching_sub_ids[0]['id'])

            sub_area_ids = filter(lambda a: a not in contained_areas, sub_area_ids)
            simpl_geoms_area = simpl_geoms.area
            diff = None
            for sub_area_id in sub_area_ids:
                sub_area_multipoly = get_area_multipolygon(sub_area_id)
                sub_area = area_sfc(sub_area_id)
                try:
                    diff = simpl_geoms.difference(sub_area_multipoly.convex_hull)
                    area_reduction = simpl_geoms_area - diff.area
                except TopologicalError:
                    try:
                        diff = geoms.difference(sub_area_multipoly)
                        area_reduction = geoms_area - diff.area
                    except TopologicalError as e:
                        error(e.message)

                if isinstance(diff, Polygon) or isinstance(diff, MultiPolygon):
                    try:
                        overlap_rate = area_reduction / sub_area
                    except ZeroDivisionError:
                        pass
                    else:
                        if overlap_rate > 0.9:
                            contained_areas.add(sub_area_id)
                            try:
                                diff_geoms = diff_geoms.difference(sub_area_multipoly)
                            except Exception:
                                for sub_area_poly in sub_area_multipoly:
                                    try:
                                        diff_geoms = diff_geoms.difference(sub_area_poly)
                                    except Exception as e:
                                        pass

    return list(contained_areas)


@cache_it_json(cache=cache_proc)
def g_area(id):
    result = api.query("""area({});out;""".format(id))
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

    debug('Sub areas of ' + area.tags['name'])
    contained_areas = query_sub_areas(id, admin_level)

    return {'type': type,
            'id': id,
            'tag': {k: v for k, v in area.tags.iteritems() if
                    not k.startswith('name') or k == 'name' or k == 'name:es' or k == 'name:es' or k == 'name:it'},
            'contains': contained_areas}


@app.route('/area/<id>')
@cache.cached(timeout=MAX_AGE, key_prefix=make_cache_key)
def get_area(id):
    area_dict = g_area(id)
    area_dict['contains'] = map(lambda a: 'area/{}'.format(a), area_dict['contains'])

    response = jsonify(area_dict)
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    response.headers['Last-Modified'] = format_date_time(time.mktime(datetime.now().timetuple()))
    return response


def simplify_factor(multipolygon, max=0.01):
    n_nodes = reduce(lambda x, y: x + len(y.exterior.coords), multipolygon, 0)
    simpl_linear_factor = 0.0000013
    simpl_factor = simpl_linear_factor * n_nodes
    if simpl_factor > max:
        simpl_factor = max
    return simpl_factor


@app.route('/area/<id>/geom')
@cache.cached(timeout=MAX_AGE)
def get_area_geom(id):
    multipolygon = get_area_multipolygon(id)
    simpl_factor = simplify_factor(multipolygon)
    multipolygon = multipolygon.simplify(simpl_factor)
    response = jsonify({'wkt': dumps(multipolygon)})
    response.headers['Cache-Control'] = 'max-age={}'.format(MAX_AGE)
    response.headers['Last-Modified'] = format_date_time(time.mktime(datetime.now().timetuple()))
    return response


def _elm_key(elm, match=set()):
    key = None
    tags = elm.get('tags', elm.get('tag'))
    matching_tags = list(set(match).intersection(set(tags.keys())))
    matching_tags = sorted(matching_tags, key=lambda x: x == 'building')
    try:
        tag = matching_tags.pop()
        key = '{}:{}'.format(tag, tags[tag])
    except IndexError:
        pass

    return key


MATCH_TAGS = {'shop', 'highway', 'amenity', 'building', 'tourism'}


@cache_it_json(cache=cache_proc)
def g_area_names():
    all_areas = api.query("""
            area[boundary=administrative][type][name][admin_level];
            out body;
            """)

    area_map = {}
    for area in all_areas.areas:
        try:
            name = area.tags.get('name', area.tags.get('name:es'))
            if name not in area_map:
                area_map[name] = []
            area_map[name].append({'id': str(area.id), 'l': int(area.tags['admin_level'])})
        except KeyError:
            pass

    return area_map


@cache_it_json(cache=cache_proc, expire=3600)
def match_area_id(name):
    area_names = g_area_names()
    match, score = process.extractOne(name, area_names.keys())
    if score > 50:
        areas = area_names[match]
        min_admin_level = min(map(lambda x: x['l'], areas))
        selected_matchings = filter(lambda x: x['l'] == min_admin_level, areas)
        final_match = selected_matchings.pop()
        return final_match['id']


@cache_it_json(cache=cache_proc)
def get_area_buildings(id):
    area_geoms = get_free_area(str(id))
    buildings = []
    for points in area_geoms:
        lat_lng_points = ['{} {}'.format(p[1], p[0]) for p in points]
        geo_filter = 'poly:"{}"'.format(' '.join(lat_lng_points))

        query_str = """            
                            way[building]["building"!~"no"]({});                            
                            out tags;
                        """.format(geo_filter)
        results = api.query(query_str, maxsize=134217728, expire=86400)
        for i, way in enumerate(results.ways):
            elm = {'id': 'way/{}'.format(way.id)}  # way_distance(osm_result, way, poi)}
            key = _elm_key({'tags': way.tags}, MATCH_TAGS)
            if key is None:
                key = 'way'
            if 'name' in way.tags:
                elm['name'] = way.tags['name']
            elm['type'] = 'way:' + key

            buildings.append(elm)
    return buildings


#@cache_it_json(cache=cache_proc)
def get_free_area(id):
    area = g_area(str(id))
    free_mp = get_area_multipolygon(id)
    if area['contains']:
        sub_mp_list = [get_area_multipolygon(sub_area) for sub_area in area['contains']]
        for smp in sub_mp_list:
            try:
                free_mp = free_mp.difference(smp)
            except TopologicalError:
                pass
        if isinstance(free_mp, Polygon):
            free_mp = MultiPolygon([free_mp])
    return [map(lambda x: tuple(x[0]), zip(p.exterior.coords)) for p in free_mp]


def area_type(admin_level):
    type = 'area'
    if admin_level == 4:
        type += ':province'
    elif admin_level == 8:
        type += ':municipality'
    elif admin_level == 2:
        type += ':country'
    return type


@app.route('/elements')
@cache.cached(timeout=MAX_AGE, key_prefix=make_cache_key)
def get_geo_elements():
    limit = request.args.get('limit', None)
    if limit:
        limit = int(limit)

    elm_filters = request.args.getlist('filter')
    elms = []

    area = request.args.get('area')

    restrict = request.args.get('restrict', None)
    restrict = restrict is not None

    if area:
        try:
            int(area)
        except ValueError:
            area = match_area_id(area)
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

    geo_filters = []

    if area:
        area_mp = get_area_multipolygon(area)
        center_tuple = list(area_mp.representative_point().coords)[0]
        center = LatLon(*reversed(center_tuple))

        free_area = get_free_area(area)
        if 'area' in elm_filters:
            area_dict = g_area(str(area))
            admin_level = int(area_dict['tag']['admin_level'])
            if restrict:
                elms = [{'type': area_type(admin_level), 'id': 'area/{}'.format(area)}]
            else:
                subareas = [g_area(sa) for sa in area_dict['contains']]
                elms = [{'type': area_type(sa['tag']['admin_level']), 'id': 'area/{}'.format(sa['id'])} for sa in
                        subareas]

        for points in free_area:
            # lat, lng = points[0][1], points[0][0]
            lat_lng_points = ['{} {}'.format(p[1], p[0]) for p in points]
            geo_filter = 'poly:"{}"'.format(' '.join(lat_lng_points))
            geo_filters.append(geo_filter)
    else:
        radius = int(request.args.get('radius', 200))
        center = LatLon(lat, lng)
        geo_filter = 'around:{},{},{}'.format(radius, lat, lng)
        geo_filters.append(geo_filter)

    for geo_filter in geo_filters:
        if elm_filters:
            way_query_union = []
            node_query_union = []
            if 'building' in elm_filters:
                way_query_union.append('way.wa[building]["building"!~"no"]')
            if 'way' in elm_filters:
                way_query_union.append('way.wa[highway]')
                way_query_union.append('way.wa[footway]')
            if 'shop' in elm_filters:
                node_query_union.append('node.na[shop]')
            if 'amenity' in elm_filters:
                node_query_union.append('node.na[amenity]')
            if 'highway' in elm_filters:
                node_query_union.append('node.na[highway]')
            if 'tourism' in elm_filters:
                node_query_union.append('node.na[tourism]')

            query_str = ''
            if way_query_union:
                query_str += 'way({}) -> .wa;\n(\n'.format(geo_filter)
                query_str += ';\n'.join(way_query_union)
                query_str += ';\n);\nout tags;'

            if node_query_union:
                query_str += 'node({}) -> .na;\n(\n'.format(geo_filter)
                query_str += ';\n'.join(node_query_union)
                query_str += ';\n);\nout tags;'
        else:
            query_str = """            
                    way[building]["building"!~"no"]({});                            
                    out tags;
                    """.format(geo_filter)

        osm_result = api.query(query_str, maxsize=134217728, expire=864000)

        for i, node in enumerate(osm_result.nodes):
            elm = {'id': 'node/{}'.format(node.id)}  # node_distance(node, poi)}
            key = _elm_key({'tags': node.tags}, MATCH_TAGS)
            if key is None:
                key = 'node'
            if 'name' in node.tags:
                elm['name'] = node.tags['name']
            elm['type'] = 'node:' + key
            if elm not in elms:
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
            if elm not in elms:
                elms.append(elm)

            if limit and i == limit - 1:
                break

    if 'area' in elm_filters:
        osm_result = api.query("""
              is_in({},{}) -> .a;
              area.a[admin_level];
              out;""".format(center.lat, center.lon))

        max_admin_level = 0
        areas_found = []
        for res_area in osm_result.areas:
            if area and restrict and int(res_area.id) != int(area):
                continue

            admin_level = int(res_area.tags['admin_level'])
            if admin_level > max_admin_level:
                max_admin_level = admin_level
            areas_found.append({'id': res_area.id, 'l': admin_level})

        for d in filter(lambda d: d['l'] == max_admin_level, areas_found):
            elm = {'id': 'area/{}'.format(d['id']), 'type': area_type(d['l'])}
            if elm not in elms:
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
