#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import datetime
import operator
import numpy as np
import pathlib
import os
import math
import typing
from datetime import datetime
from collections import deque
import progressbar
from PIL import Image
from imageio import imread


def read_xml_config(calib_filepath):
    import xml.etree.ElementTree
    e = xml.etree.ElementTree.parse(str(calib_filepath)).getroot()
    hdl_calib = e.findall("DB")[0].findall("points_")[0].findall("item")
    assert len(hdl_calib) == 64

    calib = []
    for elem in hdl_calib:
        px = elem.findall("px")[0]

        d = {}
        for entry in px.iter():
            if entry is px:
                continue
            t = entry.text
            tag = entry.tag[:-1] if entry.tag[-1] == '_' else entry.tag
            d[tag] = float(t) if '.' in t else int(t)
        calib.append(d)

    # sort according to vertical layer (from top to bottom, matching rows)
    calib = sorted(calib, key=operator.itemgetter('vertCorrection'), reverse=True)
    return calib


def read_binary_point_cloud_with_intensity(binary_pcd_file_path):
    if isinstance(binary_pcd_file_path, pathlib.PosixPath):
        binary_pcd_file_path = str(binary_pcd_file_path)
    scan = np.fromfile(binary_pcd_file_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def write_binary_point_cloud_with_intensity(binary_pcd_file_path: str, scan: np.ndarray):
    scan.tofile(binary_pcd_file_path)


def read_png_image(png_image_path):
    return np.array(imread(png_image_path))


def get_image_size(image_path):
    """Determine the image type of fhandle and return its size.
    """
    return Image.open(image_path).size


def read_timestamps_from_file(timestamps_filepath):

    if isinstance(timestamps_filepath, tuple):
        r = []
        for t in timestamps_filepath:
            r.append(read_timestamps_from_file(t))
        return tuple(r)

    stamps = []
    with open(timestamps_filepath) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) == 1:
                continue
            try:
                dt = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            except ValueError as e:
                print(line)
                print(i)
                raise e
            stamps.append(dt)
    return np.asarray(stamps)


def resize_image_with_crop_or_pad(img, target_dims):
    if img.shape[:len(target_dims)] != target_dims:
        raise NotImplementedError()
    return img


def find_jumps(pointcloud: np.ndarray, threshold: float = -.005, auto_correct=False) -> np.ndarray:
    azimuth_flipped = -np.arctan2(pointcloud[:, 1], -pointcloud[:, 0])
    jumps = np.argwhere(np.ediff1d(azimuth_flipped) < threshold)
    rows = np.zeros(shape=azimuth_flipped.shape, dtype=np.int32)
    rows[jumps + 1] = 1
    rows = np.cumsum(rows, dtype=np.int32)

    if rows[-1] < 63 and auto_correct:
        rows += (63 - rows[-1])

    return rows


def time_correct_azimuth(azimuth: np.ndarray, point_cloud: np.ndarray,
        calib: list, auto_correct_rows: bool) -> typing.Tuple[np.ndarray, np.ndarray]:

    assert point_cloud.shape[1] == 4

    # try to figure out row changes by change of y-coordinate sign
    rows = find_jumps(point_cloud, auto_correct=auto_correct_rows)

    delta_angles = np.asarray([x['rotCorrection'] * math.pi / 180.0 for x in calib])
    corrected_azimuth = azimuth - delta_angles[rows]
    return corrected_azimuth, rows


def partition_pointcloud(data: typing.Deque[typing.Tuple[np.ndarray, np.ndarray]],
                         num_sectors: int = 90)\
        -> typing.List[np.ndarray]:
    assert len(data) == 3
    bins = np.linspace(-math.pi, math.pi, num_sectors+1)
    clouds = [[] for _ in range(num_sectors)]
    for i, (a, cloud, _) in zip(range(-1, 2), data):
        if cloud is None:
            continue
        # move azimuth angles: previous scan (i == -1) is -2pi ahead, following scan +2pi late
        d = np.digitize(a + i * 2 * math.pi, bins)
        # valid sectors are at indices 1-NUM_SECTORS. 0 is previous. NUM_SECTORS+1 is following.
        for s in range(num_sectors):
            clouds[s].append(cloud[d == (s+1)])

    for i, c in enumerate(clouds):
        clouds[i] = np.concatenate(c, axis=0)
    return clouds


def partition_timestamps(dt: datetime, dt_start: datetime, dt_end: datetime, azimuth: np.ndarray,
                         num_sectors: int = 90) -> np.ndarray:
    min_a = min(azimuth)
    max_a = max(azimuth)
    a_diff = max_a - min_a
    err = (dt - dt_start) / (dt_end - dt_start) * a_diff + min_a
    m_err = 1.5e-4
    if err > m_err:
        print("Diff to target timestamp is larger {}. ({})".format(m_err, err))

    angle_interpolation_targets = np.linspace(math.pi * (1 - num_sectors) / num_sectors,
                                              math.pi * (num_sectors - 1) / num_sectors,
                                              num_sectors)
    return (angle_interpolation_targets - min_a) / a_diff * (dt_end - dt_start) + dt_start


def save_velo_data_stream(velodyne_data_folder, velodyne_target_folder,
                          velodyne_target_folder2, velo_calib):
    num_sectors = 90
    print("Exporting velodyne data")
    velo_data_dir = os.path.join(velodyne_data_folder, 'data')
    velodyne_target_folder_data = os.path.join(velodyne_target_folder, 'data')
    os.makedirs(velodyne_target_folder_data, exist_ok=True)
    velodyne_target_folder_data2 = os.path.join(velodyne_target_folder2, 'data')
    os.makedirs(velodyne_target_folder_data2, exist_ok=True)
    velo_filenames = sorted(os.listdir(velo_data_dir))

    def read_timestamps(filename='timestamps.txt') -> typing.List[datetime]:
        d = []
        with open(os.path.join(velodyne_data_folder, filename)) as f:
            lines = f.readlines()

            for line in lines:
                if len(line) == 1:
                    continue
                dt = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                d.append(dt)
        return d

    velo_datetimes = read_timestamps()
    velo_datetimes_start = read_timestamps('timestamps_start.txt')
    velo_datetimes_end = read_timestamps('timestamps_end.txt')

    if len(velo_datetimes) != len(velo_datetimes_start) != len(velo_datetimes_end):
        raise RuntimeError("Invalid datatimes len")

    data = deque(maxlen=3)
    for i in range(3):
        data.append((np.empty(shape=[0, ], dtype=np.float32), None, None))
    bar = progressbar.ProgressBar(max_value=len(velo_datetimes))

    row_counts = []
    point_cloud_counter = 0
    split_cloud_counter = 0

    timestamps_target_path = os.path.join(velodyne_target_folder, 'timestamps.txt')

    def do_stuff(counter, counter2):
        clouds = partition_pointcloud(data, num_sectors=num_sectors)
        timestamps = partition_timestamps(*data[1][2], azimuth,
                                          num_sectors=num_sectors)
        for t in timestamps:
            timestamps_file.write('{}\n'.format(t.strftime('%Y-%m-%d %H:%M:%S.%f')))

        # cut away ground (z < 1.4m)
        clouds_without_ground = [c[c[:, 2] > -1.4] for c in clouds]

        for c, c_ground in zip(clouds, clouds_without_ground):
            c.tofile(os.path.join(velodyne_target_folder_data,
                                  '{:010d}_{:02d}.bin'.format(counter, counter2)))
            c_ground.tofile(os.path.join(velodyne_target_folder_data2,
                                         '{:010d}_{:02d}.bin'.format(counter, counter2)))
            counter2 += 1

    with open(timestamps_target_path, 'w') as timestamps_file:
        for i, (dt, dt_start, dt_end, filename) in bar(enumerate(zip(velo_datetimes, velo_datetimes_start,
                                                      velo_datetimes_end, velo_filenames))):
            if dt is None:
                raise RuntimeError("Dt is None {}".format(filename))
            velo_filename = os.path.join(velo_data_dir, filename)
            # read binary data
            scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
            azimuth = np.arctan2(scan[:, 1], scan[:, 0])

            try:
                time_corrected_azimuth, row_mapping = time_correct_azimuth(azimuth, scan,
                                                                           calib=velo_calib,
                                                                           auto_correct_rows=(i != len(velo_filenames) - 1))
            except RuntimeError as e:
                raise RuntimeError("{}: {}".format(filename, str(e)))

            # _, row_count = np.unique(row_mapping, return_counts=True)
            # row_counts.append(row_count)

            data.append((time_corrected_azimuth, scan, (dt, dt_start, dt_end)))
            if data[1][1] is not None:
                do_stuff(point_cloud_counter, split_cloud_counter)
                point_cloud_counter += 1

        # process last point cloud
        data.append((np.empty(shape=[0, ], dtype=np.float32), None, None))
        do_stuff(point_cloud_counter, split_cloud_counter)
        point_cloud_counter += 1

    # copy timestamps to second location
    timestamps_second = pathlib.Path(velodyne_target_folder2) / 'timestamps.txt'
    with timestamps_second.open(mode='xb') as tid:
        tid.write(pathlib.Path(timestamps_target_path).read_bytes())
