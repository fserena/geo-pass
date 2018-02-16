import calendar
import json
import math
import multiprocessing
import re
from Queue import Empty
from datetime import datetime
from multiprocessing import Lock, Queue
from random import sample
from time import sleep

import requests
from concurrent.futures import ThreadPoolExecutor
from dateutil import parser
from requests.utils import parse_dict_header

from geo_pass import geocoding

lock = Lock()

workers = (multiprocessing.cpu_count() * 2) + 1


def extract_ids(data):
    if data:
        data_str = json.dumps(data)
        res = re.findall(r'\w+\/\d+', data_str)
        filtered_res = [r for r in res if r.split('/')[0] in ['node', 'way', 'area']]
        return sample(filtered_res, k=len(filtered_res))


def extract_ttl(headers):
    ttl = None
    cache_control = headers.get('Cache-Control', None)
    if cache_control is not None:
        cache_dict = parse_dict_header(cache_control)
        max_age = cache_dict.get('max-age', ttl)
        if max_age is not None:
            ttl = int(math.ceil(float(max_age)))
    return ttl


def get(id, buffer, helpers, follow):
    helpers['pool'].submit(_get, id, buffer, helpers, follow)


def _get(id, buffer, helpers, follow):
    if helpers['queue']._closed:
        return

    now = calendar.timegm(datetime.utcnow().timetuple())

    with helpers['lock']:
        last_ts = helpers['context']['1']
        elapsed = now - last_ts

    # print elapsed

    sleep(max(0.0, 1000.0 - elapsed) / 1000.0)

    url = 'http://127.0.0.1:5006/{}'.format(id)
    if id == 'elements' or buffer['radius']:
        url += '?lat={}&lng={}'.format(buffer['lat'],
                                       buffer['lng'])

    if buffer['radius']:
        url += '&radius={}'.format(buffer['radius'])

    print 'getting', url
    helpers['idle'] = False
    pre_req = datetime.now()
    response = requests.get(url)
    post_req = datetime.now()
    helpers['idle'] = True

    ttl = extract_ttl(response.headers)
    last_modified_str = response.headers.get('Last-Modified')
    date_str = response.headers.get('Date')
    last_modified = parser.parse(last_modified_str)
    get_date = parser.parse(date_str)
    if (get_date - last_modified).total_seconds() < 1:
        helpers['context']['1'] = calendar.timegm(datetime.utcnow().timetuple())

    helpers['context']['2'] = (post_req - pre_req).total_seconds()

    if response.status_code == 200:
        data = response.json()
        follow(data, buffer, helpers)


def follow_element(data, buffer, helpers):
    helpers['queue'].put(data)
    shuffled_res = extract_ids(data)
    if shuffled_res:
        for r in shuffled_res:
            # if random() > 0.5:
            helpers['pool'].submit(get_element, r, buffer, helpers)


def get_element(id, buffer, helpers):
    if not helpers['queue']._closed:
        try:
            do_get = False
            with helpers['lock']:
                if id not in helpers['trace']:
                    do_get = True
                    helpers['trace'].append(id)

            if do_get:
                get(id, buffer, helpers, follow_element)
        except Exception as e:
            print e.message


def follow_elements(data, buffer, helpers):
    shuffled_res = extract_ids(data)
    for id in shuffled_res:
        helpers['pool'].submit(get_element, id, buffer, helpers)


def get_elements(buffer, helpers):
    get('elements', buffer, helpers, follow_elements)


class Crawler(object):
    def __init__(self, location=None, lat=None, lng=None, radius=None, pool=None):
        if location is not None:
            ll = None
            while ll is None:
                ll = geocoding(location)
                if not ll:
                    sleep(1)
            lat, lng = ll['lat'], ll['lng']

        self.lat = lat
        self.lng = lng
        self.radius = radius
        self.pool = pool

    def __iter__(self):
        helpers = crawl_from(self.lat, self.lng, self.radius, pool=self.pool)
        queue = helpers.get('queue')
        pool = helpers.get('pool')
        retry = 0

        while True:
            try:
                element = queue.get(timeout=1)
                retry = 0
                yield element
            except Empty as e:
                if helpers['idle'] and not len(pool._work_queue.queue):
                    if retry < 5:
                        retry += 1
                    else:
                        raise StopIteration


def crawl_from(lat, lng, radius=None, pool=None):
    context = {
        '1': calendar.timegm(datetime.min.timetuple()),
        '2': 10
    }

    trace = []

    if pool is None:
        pool = ThreadPoolExecutor(max_workers=2)
    queue = Queue(maxsize=10)
    # buffer = {'lat': lat, 'lng': lng, 'radius': radius}
    helpers = {'context': context, 'trace': trace, 'lock': lock, 'pool': pool, 'queue': queue, 'idle': False}

    # lat + (i - workers / 2) * 0.0001,
    # lng + (i - workers / 2) * 0.0001,

    pool.submit(get_elements, {
        'lat': lat,
        'lng': lng,
        'radius': radius
    }, helpers)

    # [pool.submit(get_elements, {
    #     'lat': lat + (i - workers / 2) * 0.0001,
    #     'lng': lng + (i - workers / 2) * 0.0001,
    #     'radius': radius
    # }, helpers) for i in range(workers)]

    return helpers

# if __name__ == '__main__':
# freeze_support()
# manager = Manager()
# d = manager.dict()
# l = manager.list()
# queue = manager.Queue()
# lock = Lock()
# pool = ThreadPoolExecutor(max_workers=2)

# ll = None
# while ll is None:
#     ll = geocoding(location)
#     if not ll:
#         sleep(1)
# lat, lng = ll['lat'], ll['lng']

# th_pool = ThreadPoolExecutor(max_workers=workers)

# futures = [th_pool.submit(get_elements,
#                           lat + (i - workers / 2) * 0.001,
#                           lng + (i - workers / 2) * 0.001,
#                           pool, lock, l, d) for i in range(workers)]

# wait(futures)
