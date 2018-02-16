# coding=utf-8
from py2neo import Graph, Node, Relationship

from crawler import Crawler

g = Graph(host='127.0.0.1', password='XXXX')

g.delete_all()


def add_way(g, data):
    osm_id = unicode(data['id'])
    way_name = unicode(data['tag'].get('name', ''))
    if way_name:
        del data['tag']['name']

    n = g.find_one("Way", property_key='osm_id', property_value=osm_id)
    if n is None:
        n = Node("Way", name=way_name, osm_id=osm_id, api_path='way/{}'.format(osm_id), **data['tag'])
    else:
        n['name'] = way_name
        for k, v in data['tag'].items():
            n[k] = v
        g.push(n)

    for bid in data.get('buildings', []):
        bid = bid.split('/')[1]
        b = g.find_one("Building", property_key='osm_id', property_value=str(bid))
        if b is None:
            b = Node("Building", osm_id=str(bid), api_path='way/{}'.format(bid))
            g.create(b)

        nb = Relationship(n, "GOES_BY", b)
        g.create(nb)

    for wid in data.get('intersect', []):
        wid = wid.split('/')[1]
        w = g.find_one("Way", property_key='osm_id', property_value=str(wid))
        if w is None:
            w = Node("Way", osm_id=str(wid), api_path='way/{}'.format(wid))
            g.create(w)
        nw = Relationship(n, "INTERSECTS", w)
        g.create(nw)

    for nid in data.get('contains', []):
        nid = nid.split('/')[1]
        ne = g.find_one("Node", property_key='osm_id', property_value=str(nid))
        if n is None:
            n = Node("Node", osm_id=str(bid), api_path='node/{}'.format(nid))
            g.create(n)

        nb = Relationship(n, "GOES_BY", b)
        g.create(nb)

    return n


def add_node(g, data):
    latitude = float(data['lat'])
    longitude = float(data['lon'])
    n = g.find_one("Node", property_key='osm_id', property_value=str(data['id']))
    if n is None:
        ptypes = data['type'].split(':')
        ptypes = [t.title() for t in ptypes]
        n = Node("Node", *ptypes,
                 osm_id=str(data['id']),
                 api_path='node/{}'.format(data['id']),
                 latitude=latitude,
                 longitude=longitude, **data['tag'])
        g.create(n)
    else:
        n['latitude'] = latitude
        n['longitude'] = longitude
        for k, v in data['tag'].items():
            n[k] = v
        g.push(n)

    for aid in data.get('areas', []):
        aid = aid.split('/')[1]
        a = g.find_one("Area", property_key='osm_id', property_value=str(aid))
        if a is None:
            a = Node("Area", osm_id=str(aid), api_path='area/{}'.format(aid))
            g.create(a)

        na = Relationship(n, "IS_IN", a)
        g.create(na)


def add_area(g, data):
    a = g.find_one("Area", property_key='osm_id', property_value=str(data['id']))
    if a is None:
        ptypes = data['type'].split(':')
        ptypes = [t.title() for t in ptypes]
        n = Node(*ptypes,
                 osm_id=str(data['id']),
                 api_path='area/{}'.format(data['id']), **data['tags'])
        g.create(n)
    else:
        for k, v in data['tags'].items():
            a[k] = v
        g.push(a)


def add_building(g, data):
    latitude = float(data['center']['lat'])
    longitude = float(data['center']['lon'])
    n = g.find_one("Building", property_key='osm_id', property_value=str(data['id']))
    if n is None:
        n = Node("Building",
                 osm_id=str(data['id']),
                 api_path='way/{}'.format(data['id']),
                 latitude=latitude,
                 longitude=longitude, **data['tag'])
        g.create(n)
    else:
        n['latitude'] = latitude
        n['longitude'] = longitude
        for k, v in data['tag'].items():
            n[k] = v
        g.push(n)

    for wid in data.get('ways', []):
        wid = wid.split('/')[1]
        w = g.find_one("Way", property_key='osm_id', property_value=str(wid))
        if w is None:
            w = Node("Way", osm_id=str(wid), api_path='way/{}'.format(wid))
            g.create(w)

        nw = Relationship(w, "GOES_BY", n)
        g.create(nw)

    for bid in data.get('surrounding_buildings', []):
        bid = bid.split('/')[1]
        b = g.find_one("Building", property_key='osm_id', property_value=str(bid))
        if b is None:
            b = Node("Building", osm_id=str(bid), api_path='way/{}'.format(bid))
            g.create(b)

        # sub_graph |= b
        nb = Relationship(n, "ADJACENT_TO", b)
        g.create(nb)
        # sub_graph |= nb

    for pdict in data.get('contains', []):
        pid = pdict['id']
        ptypes = pdict['type'].split(':')
        ptypes = [t.title() for t in ptypes]
        pid = pid.split('/')[1]
        p = g.find_one("Node", property_key='osm_id', property_value=str(pid))
        if p is None:
            p = Node("Node", *ptypes, osm_id=str(pid), api_path='node/{}'.format(pid))
            g.create(p)

        np = Relationship(n, "CONTAINS", p)
        g.create(np)

    for aid in data.get('areas', []):
        aid = aid.split('/')[1]
        a = g.find_one("Area", property_key='osm_id', property_value=str(aid))
        if a is None:
            a = Node("Area", osm_id=str(aid), api_path='area/{}'.format(aid))
            g.create(a)

        na = Relationship(n, "IS_IN", a)
        g.create(na)

    return n


def is_way(element):
    return element.get('type', '') == 'way'


def is_building(element):
    return element.get('type', '') == 'building'


def is_area(element):
    return element.get('type', '') == 'area'


def push(element):
    try:
        if 'id' in element:
            if is_way(element):
                add_way(g, element)
            elif is_building(element):
                add_building(g, element)
            elif is_area(element):
                add_area(g, element)
            else:
                add_node(g, element)
    except Exception, e:
        print e.message


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor

    th = ThreadPoolExecutor(max_workers=1)
    crawler = Crawler(location=u"plaza del este madrid", radius=80, pool=th)

    for elm in crawler:
        push(elm)
        print elm
