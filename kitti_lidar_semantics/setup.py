#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    'Click>=6.0',
    'numpy',
    'imageio',
    'scipy',
    'pykitti',
    'opencv-python',
    'progressbar2',
]

setup_requirements = [
    'pytest-runner',
]

setup(
    name='kitti_lidar_semantics',
    version='0.1.0',
    description="Annotated LiDAR measurements with semantic classes from RGB images.",
    author="Christoph Rist",
    author_email='c.rist@posteo.de',
    packages=find_packages(include=['kitti_lidar_semantics']),
    entry_points={
        'console_scripts': [
            'run_on_kitti=kitti_lidar_semantics.run_on_kitti:main',
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    setup_requires=setup_requirements,
)
