#!/bin/bash

cat installed_files.txt | xargs rm -rf
python setup.py install --record installed_files.txt
