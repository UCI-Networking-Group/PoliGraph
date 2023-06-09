#!/usr/bin/env python

from setuptools import setup

setup(
    name='poligrapher',
    author='UCI Networking Group',
    include_package_data=True,
    packages=['poligrapher', 'poligrapher.annotators', 'poligrapher.scripts'],
)
