#!/bin/sh
echo "Starting geopass..."

/root/.env/bin/pip install --upgrade pip
/root/.env/bin/pip install --upgrade git+https://github.com/fserena/geo-pass.git
/root/.env/bin/geo-pass &
