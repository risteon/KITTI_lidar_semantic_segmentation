#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='read_transform',
    version='1.0',
    description='Read transforms from rosbag and save to txt',
    author='Christoph Rist',
    author_email='c.rist@posteo.de',
    entry_points = {
        'console_scripts': ['read_transform_from_rosbag=read_transform.read_transform_from_rosbag:main'],
    },
    install_requires=['click',],
    include_package_data=True,
    packages=find_packages(include=['read_transform']),
)
