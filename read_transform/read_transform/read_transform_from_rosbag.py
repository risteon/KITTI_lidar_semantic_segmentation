#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ROS NEEDS PYTHON2.7
Open rosbag from cartographer, read tf topics and save in useful format.
"""
from __future__ import print_function

import rosbag
import datetime
import click


def read_rosbag(bagfile, output_timestamps, output_poses):
    bag = rosbag.Bag(bagfile)

    for topic, msg, t in bag.read_messages(topics=['/tf']):
        assert topic == '/tf'
        assert len(msg.transforms) == 1
        assert msg.transforms[0].header.frame_id == 'map'
        # timestamp
        nsecs = t.to_nsec()
        s = datetime.datetime.fromtimestamp(nsecs // int(1e9)).strftime(
            '%Y-%m-%d %H:%M:%S') + '.' + str(int(nsecs % int(1e9))).zfill(9)
        output_timestamps.write('{}\n'.format(s))
        # transform
        transform = msg.transforms[0].transform
        output_poses.write('trans {} rot {}\n'.format(
            ';'.join(str(getattr(transform.translation, x)) for x in ['x', 'y', 'z']),
            ';'.join(str(getattr(transform.rotation, x)) for x in ['w', 'x', 'y', 'z'])))

    bag.close()


@click.command()
@click.argument('bagfile', nargs=1, type=click.Path(exists=True))
@click.argument('output_timestamps', nargs=1, type=click.File('wb'))
@click.argument('output_poses', nargs=1, type=click.File('wb'))
def main(bagfile, output_timestamps, output_poses):
    read_rosbag(bagfile, output_timestamps, output_poses)
