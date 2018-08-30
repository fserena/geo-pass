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

import requests

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
