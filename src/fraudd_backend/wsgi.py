#!/bin/env python3
from fraudd import create_instance
application = app = create_instance()


"""
# How to run the app
export NG_ENDPOINTS="10.1.1.168:9669";
export FLASK_ENV=development;
export FLASK_APP=wsgi;
python3 -m flask run --reload
"""
