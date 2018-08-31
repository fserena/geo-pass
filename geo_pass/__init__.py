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
import logging
import os
import zlib

import requests
from redis_cache import SimpleCache, to_unicode, json

__author__ = 'Fernando Serena'

log_level = int(os.environ.get('LOG_LEVEL', logging.INFO))

log = logging.getLogger('geopass')
log.setLevel(log_level)
for h in log.handlers:
    log.removeHandler(h)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setLevel(log_level)
ch.setFormatter(formatter)
log.addHandler(ch)

google_api_key = os.environ.get('GOOGLE_API_KEY', None)


def geocoding(address):
    response = requests.get(
        u'https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}'.format(
            address, google_api_key))

    if response.status_code == 200:
        try:
            result = response.json().get('results').pop()
            location = result['geometry']['location']
            # print 'geocoding done.'
            return location
        except IndexError:
            pass


def streetview(lat, lng):
    response = requests.get(
        u'https://maps.googleapis.com/maps/api/streetview?size=600x300&location={},{}&key={}'.format(
            lat, lng, google_api_key))

    if response.status_code == 200:
        return response.content


def debug(msg):
    try:
        log.debug(msg)
    except Exception:
        pass


def info(msg):
    try:
        log.info(msg)
    except Exception:
        pass


def error(msg):
    try:
        log.error(msg)
    except Exception:
        pass


class ZSimpleCache(SimpleCache):
    def make_key(self, key):
        return "C-{0}:{1}".format(self.prefix, key)

    def get_set_name(self):
        return "C-{0}-keys".format(self.prefix)

    def store_json(self, key, value, expire=None):
        if 'osm3s' in value:
            del value['osm3s']
            del value['version']
            del value['generator']
        ss = json.dumps(value)
        value = zlib.compress(ss)
        self.store(key, value, expire)

    def get_json(self, key):
        return json.loads(zlib.decompress(self.get(key)))

    def mget_json(self, keys):
        """
        Method returns a dict of key/values for found keys with each value
        parsed from JSON format.
        :param keys: array of keys to look up in Redis
        :return: dict of found key/values with values parsed from JSON format
        """
        d = self.mget(keys)
        if d:
            for key in d.keys():
                d[key] = json.loads(zlib.decompress(d[key])) if d[key] else None
            return d

    def store(self, key, value, expire=None):
        """
        Method stores a value after checking for space constraints and
        freeing up space if required.
        :param key: key by which to reference datum being stored in Redis
        :param value: actual value being stored under this key
        :param expire: time-to-live (ttl) for this datum
        """
        key = to_unicode(key)
        # value = to_unicode(value)
        set_name = self.get_set_name()

        while self.connection.scard(set_name) >= self.limit:
            del_key = self.connection.spop(set_name)
            self.connection.delete(self.make_key(del_key))

        pipe = self.connection.pipeline()
        if expire is None:
            expire = self.expire

        if isinstance(expire, int) and expire <= 0:
            pipe.set(self.make_key(key), value)
        else:
            pipe.setex(self.make_key(key), expire, value)

        pipe.sadd(set_name, key)
        pipe.execute()
