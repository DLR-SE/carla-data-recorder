#!/bin/bash

CDR_THREADED=1 coverage run -m pytest -v
coverage html --show-contexts
