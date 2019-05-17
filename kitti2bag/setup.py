#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='kitti2bag',
    version='1.5',
    description='Convert KITTI dataset to ROS bag file the easy way!',
    author='Tomas Krejci',
    author_email='tomas@krej.ci',
    url='https://github.com/tomas789/kitti2bag/',
    download_url = 'https://github.com/tomas789/kitti2bag/archive/1.5.zip',
    keywords = ['dataset', 'ros', 'rosbag', 'kitti'],
    entry_points = {
        'console_scripts': ['kitti2bag=kitti2bag.__main__:main'],
    },
    install_requires=['pykitti', 'progressbar2'],
    include_package_data=True,
    packages=find_packages(include=['kitti2bag']),
)
