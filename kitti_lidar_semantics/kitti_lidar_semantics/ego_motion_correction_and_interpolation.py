#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pathlib
import collections
import os
import datetime
import pykitti
import numpy as np
import typing
import progressbar
import math
import tensorflow as tf
import logging
# SciPy
import scipy.interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from .utils import \
    read_binary_point_cloud_with_intensity, \
    read_timestamps_from_file, \
    read_png_image, \
    get_image_size, \
    write_binary_point_cloud_with_intensity, \
    onehot_initialization, \
    Label, \
    labels
from .utils import find_jumps


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


Data = collections.namedtuple('Data', field_names=['timestamps', 'data', 'file_extension'])

T_START = 'trans '
R_START = ' rot '
NUM_SECTORS_PER_REV = 90

train_labels = sorted([x for x in labels if x.ignoreInEval is False],
                      key=lambda x: x.trainId)
train_colors = np.stack(x.color for x in train_labels).astype(np.uint8)

DEBUG = True


def make_debug_image(image: np.ndarray, img_coords: np.ndarray, valid_mask: np.ndarray,
                     point_cloud: np.ndarray, per_image_probs: np.ndarray,
                     valid_mapping: np.ndarray, sem_image: np.ndarray = None):
    from PIL import Image
    import colorsys

    def make_color(d):
        return tuple(int(round(255.0 * v))
                     for v in colorsys.hsv_to_rgb((d % 20.0) / 20.0, 1.0, 1.0))

    # original coordinates
    # xyz = point_cloud[:, :3][valid_mask]
    # dist = np.linalg.norm(xyz, axis=-1)
    disp = img_coords[valid_mask]

    img = Image.fromarray(image)
    img = img.resize(tuple(x * 2 for x in img.size))

    # semantic coloring, merge images
    per_image_probs = np.mean(per_image_probs, axis=0)
    point_colors = np.round(np.matmul(per_image_probs, train_colors)).astype(np.uint8)

    # blend if semantic segmentation is available
    if sem_image is not None:
        if sem_image.shape != image.shape:
            raise RuntimeError("Image shapes do not match.")
        img_sem = Image.fromarray(sem_image)
        img_sem = img_sem.resize(tuple(x * 2 for x in img_sem.size))

        img = img.convert('RGBA')
        img_sem = img_sem.convert('RGBA')
        img = Image.blend(img, img_sem, alpha=0.6)

    # draw LiDAR measurments
    px = img.load()
    for uv, c in zip(disp, point_colors):
        # do not catch IndexError
        # color using distance
        # px[int(uv[0] * 2.0), int(uv[1] * 2.0)] = make_color(d)
        # color using probs
        px[int(uv[0] * 2.0), int(uv[1] * 2.0)] = tuple(c)

    return img


def project_onto_image(point_cloud: np.ndarray, image_shape: tuple,
                       t_cam_velo: np.ndarray, p_cam_img: np.ndarray, r0_rot: np.ndarray):
    """

    """

    assert len(image_shape) == 2
    homogeneous = point_cloud.copy()
    homogeneous[:, 3] = 1.0
    # N x 4 x 1
    homogeneous = np.expand_dims(homogeneous, axis=-1)

    t = np.matmul(np.matmul(p_cam_img, r0_rot), t_cam_velo)
    # t = np.matmul(p_cam_img, t_cam_velo)
    image_coordinates = np.squeeze(np.matmul(t, homogeneous),
                                   axis=-1)

    image_u = (image_coordinates[:, 0] / image_coordinates[:, 2])
    image_v = (image_coordinates[:, 1] / image_coordinates[:, 2])
    # valid if z > 0 and within image plane
    valid_mask = np.all(np.stack((image_coordinates[:, 2] > 0.0,
                                  image_v >= 0.0, image_v < image_shape[0],
                                  image_u >= 0.0, image_u < image_shape[1])), axis=0)

    img_coords = np.stack((image_u, image_v), axis=-1)
    # set points out of image to NaN
    img_coords[~valid_mask] = np.nan
    return img_coords, valid_mask


def interpolate(targets: np.ndarray, data: np.ndarray, target_size: np.ndarray):
    # data is [Channel, Height, Width]
    x = np.linspace(0.5, target_size[0] - 0.5, data.shape[1])
    y = np.linspace(0.5, target_size[1] - 0.5, data.shape[2])

    output = np.empty(shape=[data.shape[0], targets.shape[0]], dtype=data.dtype)

    for i, channel in enumerate(data):
        f = scipy.interpolate.RectBivariateSpline(x, y, channel, kx=1, ky=1)
        output[i, :] = f(targets[:, 1], targets[:, 0], grid=False)
    return output


def fuse(interpolated: np.ndarray):
    """

    :param interpolated: [ #images x #classifiers x #classes x #points]
    :return: [ #classes x #points]
    """
    # mean fusion over all images and classifiers
    return np.mean(interpolated, axis=(0, 1))


def calc_image_consensus(interpolated: np.ndarray):
    n_points = interpolated.shape[-1]

    # per image mean fusion
    per_image_fused = np.mean(interpolated, axis=1)

    # voting fusion
    one_hot = onehot_initialization(np.argmax(interpolated.transpose([0, 1, 3, 2]), axis=-1),
                                    ncols=interpolated.shape[2])
    votes_per_image = np.sum(one_hot, axis=1, dtype=one_hot.dtype)
    votes_overall = np.sum(one_hot, axis=(0, 1), dtype=one_hot.dtype)

    #
    cls_mean_fusion = np.argmax(per_image_fused, axis=1)
    cls_vote_fusion = np.argmax(votes_per_image, axis=-1)

    def get_image_consensus(cls):
        same_cls = np.all(cls == cls[0, :], axis=0)
        return np.count_nonzero(same_cls) / n_points, same_cls

    cls_image_consensus_mean, mask_mean_consens = get_image_consensus(cls_mean_fusion)
    cls_image_consensus_vote, _ = get_image_consensus(cls_vote_fusion)
    cls = np.argmax(votes_overall, axis=-1)
    cls_votes = np.max(votes_overall, axis=-1)
    return cls, cls_votes, cls_image_consensus_mean, cls_image_consensus_vote, mask_mean_consens


def single_frame_stats(interpolated: np.ndarray, scales):
    # [img2/0.7 img2/1.0 img2/1.2 img3/0.7 img3/1.0 img3/1.2]
    assert interpolated.shape[1] == len(scales)
    res = calc_image_consensus(interpolated)
    return res


def interpolate_and_fuse(image_coords: typing.List[np.ndarray],
                         valid_mask: np.ndarray,
                         sem_probs: typing.List[typing.Dict[str, np.ndarray]],
                         scales, channels_last=True):
    assert len(image_coords) == len(sem_probs)

    if channels_last:
        probs = [{k: v.transpose([2, 0, 1]) for k, v in sp.items() if k in scales}
                 for sp in sem_probs]
    else:
        probs = [{k: v for k, v in sp.items() if k in scales} for sp in sem_probs]

    # map all valid points to index in original point cloud
    valid_mapping = np.arange(len(valid_mask), dtype=np.int32)[valid_mask]

    num_channels = next(iter(probs[0].values())).shape[0]
    data_dtype = next(iter(probs[0].values())).dtype
    results = np.empty(shape=[len(image_coords), len(scales), num_channels,
                              valid_mapping.shape[0]], dtype=data_dtype)

    # i: index over images (stereo), j: index over prediction scales
    for i, (coords, prob, sem) in enumerate(zip(image_coords, probs, sem_probs)):
        valid_coords = coords[valid_mask]
        for j, s in enumerate(scales):
            # use actual output size if available, else target size
            target_size = sem.get('output_size', sem['target_size'])
            results[i, j, ...] = interpolate(targets=valid_coords, data=prob[str(s)],
                                             target_size=target_size)

    return results, valid_mapping


def get_files(data_src, *args) -> typing.Dict[str, np.ndarray]:
    d = {
        k: np.asarray([data_src[k].data / x for x in sorted(os.listdir(data_src[k].data))
                       if x.endswith(data_src[k].file_extension)])
        for k in args
    }
    return d


def calc_sensor_head_azimuth(point_cloud: np.ndarray,
        calib: list, auto_correct: bool) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    This function calculates the position of the velodyne HDL-64 sensorhead for every point in a
    given point cloud. This is based on a calibration offset to the azimuth angle (Todo?)
    """
    assert point_cloud.shape[1] == 4
    azimuth = np.arctan2(point_cloud[:, 1], point_cloud[:, 0])
    # try to figure out row changes by change of y-coordinate sign
    # 'rows' is the sensor row of every point
    rows = find_jumps(point_cloud, auto_correct=auto_correct)

    delta_angles = np.asarray([x['rotCorrection'] * math.pi / 180.0 for x in calib])
    corrected_azimuth = azimuth - delta_angles[rows]
    return corrected_azimuth, rows


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


def correct_ego_motion(point_cloud: np.ndarray, poses_interpolator, t, t_start, t_end,
        velo_calib, zero_time, auto_correct: bool):

    sensorhead_azimuth, row_mapping = calc_sensor_head_azimuth(point_cloud, calib=velo_calib,
                                                               auto_correct=auto_correct)
    # 1) create bins to include every point (more than 360 degrees necessary because of correction)
    num_sectors_with_overlap = NUM_SECTORS_PER_REV + NUM_SECTORS_PER_REV // 8 + 2
    total_angle_with_overlap = num_sectors_with_overlap / NUM_SECTORS_PER_REV * math.pi
    bin_center_begin = total_angle_with_overlap - math.pi / NUM_SECTORS_PER_REV
    bins = np.linspace(-total_angle_with_overlap, total_angle_with_overlap,
                       num_sectors_with_overlap + 1)
    bin_center = np.linspace(-bin_center_begin, bin_center_begin, num_sectors_with_overlap)
    # 2) calc timestamp for each bin
    # Todo:   there is a small time difference between 'sensor at 0 degrees'
    # Todo:   and the start/stop timestamp
    delta_secs = np.asarray([datetime.timedelta.total_seconds(t_start - zero_time),
                             datetime.timedelta.total_seconds(t_end - zero_time)])
    secs_for_bins = scipy.interpolate.interp1d(
        np.asarray([-math.pi, math.pi]), delta_secs,
        kind='linear', axis=0, copy=False, fill_value='extrapolate', assume_sorted=True)(bin_center)
    # sensor poses for each bin
    sensor_poses = poses_interpolator(secs_for_bins)
    # sensor pose for t
    pose_base = poses_interpolator(np.asarray([datetime.timedelta.total_seconds(t - zero_time)]))[0]
    # sensor poses in coordinate system of pose_base
    poses = np.matmul(np.linalg.inv(pose_base), sensor_poses)

    # get index of bin for every point in point cloud
    d = np.searchsorted(bins, sensorhead_azimuth, side='left')

    ego_motion_corrected = np.empty(shape=point_cloud.shape, dtype=point_cloud.dtype)
    p = 0
    # perform ego motion correction for every bin. Keep ordering of points!
    for s in range(num_sectors_with_overlap):
        sector = point_cloud[d == (s + 1)]
        points = sector.copy()
        points[:, 3] = 1.0
        transformed = np.squeeze(np.matmul(poses[s], np.expand_dims(points, axis=-1)), axis=-1)
        # ego_motion_corrected[p:p + len(points), ...] = \
        #     np.squeeze(np.matmul(poses[s], np.expand_dims(points, axis=-1)), axis=-1)
        # ego_motion_corrected[p:p + len(points), 3] = sector[:, 3]
        ego_motion_corrected[d == (s + 1)] = \
            np.concatenate((transformed[:, :3], sector[:, 3:]), axis=-1)
        p += len(points)

    assert p == len(point_cloud) and point_cloud.shape == ego_motion_corrected.shape
    return ego_motion_corrected, pose_base


def load_poses_from_file(filepath_poses: pathlib.Path):

    with open(str(filepath_poses)) as f:
        lines = f.readlines()
        # Format is [N poses x (translation: xyz, rotation: wxyz)] (scalar first quaternion)
        poses = np.empty(shape=[len(lines), 7], dtype=np.float32)
        for i, line in enumerate(lines):
            t_pos = line.find(T_START)
            r_pos = line.find(R_START)

            if t_pos < 0 or r_pos < 0:
                raise ValueError("Invalid poses line {}.".format(line))

            poses[i, :3] = np.asarray(list(map(float, line[t_pos + len(T_START):r_pos].split(';'))))
            poses[i, 3:] = np.asarray(list(map(float, line[r_pos + len(R_START):].split(';'))))

    return poses


def prepare_ego_motion_interpolation(data_src):
    """

    :return: Tuple of:
        A function that takes the target time as argument and returns the interpolated pose
        and the zero timestamp
    """
    poses = load_poses_from_file(data_src['map_cartographer'].data)
    poses_datetimes = read_timestamps_from_file(data_src['map_cartographer'].timestamps)
    # scipy uses scalar last xyzw quaternion format. So reorder from 'wxyz' to 'xyzw'
    rotations = R.from_quat(poses[:, [4, 5, 6, 3]], normalized=False)
    delta_sec = np.vectorize(datetime.timedelta.total_seconds)(poses_datetimes - poses_datetimes[0])
    slerp = Slerp(delta_sec, rotations)
    transl = scipy.interpolate.interp1d(delta_sec, poses[:, 0:3], kind='linear', axis=0, copy=False,
                                        fill_value='extrapolate', assume_sorted=True)

    def extrapolate_constant(func):
        def extr(t: np.ndarray):
            return func(np.where(t > delta_sec[-1], delta_sec[-1],
                                 np.where(t < delta_sec[0], delta_sec[0], t)))
        return extr

    slerp = extrapolate_constant(slerp)

    def get_poses_at(ts: np.ndarray):
        x = np.concatenate((slerp(ts).as_dcm(), np.expand_dims(transl(ts), axis=2)), axis=2)
        h = np.zeros(shape=[x.shape[0], 1, 4])
        h[:, :, 3] = 1.0
        return np.concatenate((x, h), axis=1)
    return get_poses_at, poses_datetimes[0]


def process_single_example(i, path_pointcloud, path_img_02, path_img_03, path_sem_02, path_sem_03,
                           path_sem_img_02, path_sem_img_03,
                           scales, poses_interpolator, kitti, timestamps, velo_calib, zero_time,
                           enable_ego_motion_correction=True, channels_last=True,
                           last_example=False):
    point_cloud = read_binary_point_cloud_with_intensity(path_pointcloud)
    image_02_shape = get_image_size(path_img_02)[::-1]
    image_03_shape = get_image_size(path_img_03)[::-1]

    if not enable_ego_motion_correction:
        ego_motion_corrected = point_cloud
    else:
        try:
            ego_motion_corrected, sensor_pose = correct_ego_motion(
                point_cloud, poses_interpolator,
                *timestamps,
                velo_calib, zero_time,
                auto_correct=not last_example
            )
        except RuntimeError as e:
            raise RuntimeError("{}: {}".format(path_pointcloud, str(e)))

    for x, desc in ((ego_motion_corrected, 'corrected'), ):  # (point_cloud, 'original')):
        #
        img_coords_02, valid_mask_02 = project_onto_image(x, image_02_shape,
                                                          kitti.calib.T_cam0_velo,
                                                          kitti.calib.P_rect_20,
                                                          kitti.calib.R_rect_00)
        img_coords_03, valid_mask_03 = project_onto_image(x, image_03_shape,
                                                          kitti.calib.T_cam0_velo,
                                                          kitti.calib.P_rect_30,
                                                          kitti.calib.R_rect_00)

        # only use points visible in both images
        valid_mask = np.logical_and(valid_mask_02, valid_mask_03)

        # load semantic output from files
        sem_probs = [
            dict(np.load(path_sem_02)),
            dict(np.load(path_sem_03)),
        ]
        # RETURNS: [images x scales x channels x points]
        interpolated_probs, valid_mapping = interpolate_and_fuse(
            [img_coords_02, img_coords_03], valid_mask, sem_probs, scales, channels_last)
        # RESULTS: [images x points x channels]
        per_image_probs = np.mean(interpolated_probs, axis=(1, )).transpose((0, 2, 1))

        stats = single_frame_stats(interpolated_probs, scales)

        # project onto RGB image (for visual inspection)
        if i % 50 == 0 or last_example:
            print("Consensus Mean: {}, Consensus Vote: {}".format(stats[2], stats[3]))
            # stats[4] is concensus mask
            # valid_mask[valid_mask] = stats[4]

            image_02 = read_png_image(path_img_02)
            sem_img_02 = read_png_image(path_sem_img_02)
            debug_img_02 = make_debug_image(image_02, img_coords_02, valid_mask, x,
                                            sem_image=sem_img_02,
                                            per_image_probs=per_image_probs,
                                            valid_mapping=valid_mapping)
            image_03 = read_png_image(path_img_03)
            sem_img_03 = read_png_image(path_sem_img_03)
            debug_img_03 = make_debug_image(image_03, img_coords_03, valid_mask, x,
                                            sem_image=sem_img_03,
                                            per_image_probs=per_image_probs,
                                            valid_mapping=valid_mapping)
            debug_img = debug_img_02, debug_img_03
        else:
            debug_img = None

    return ego_motion_corrected, per_image_probs, valid_mapping, sensor_pose, stats, debug_img


def make_example_proto(data_probs: np.ndarray, valid_mapping: np.ndarray) -> tf.train.Example:

    def _float_array_feature(array):
        return tf.train.Feature(float_list=tf.train.FloatList(value=array))

    def _int_array_feature(array):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=array))

    feature_dict = {
        'probs_data': _float_array_feature(data_probs.flatten()),
        'probs_shape': _int_array_feature(np.asarray(data_probs.shape, dtype=np.int64)),
        'probs_mapping': _int_array_feature(valid_mapping.astype(np.int64)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def do_work(kitti, data_src, data_target, velo_calib, scales,
            enable_ego_motion_correction=True, progress_bar=True, channels_last=True,
            missing_files=None):
    """
    Process a single KITTI RAW sequence
    :param kitti:
    :param data_src:
    :param velo_calib:
    :param enable_ego_motion_correction:
    :param progress_bar:
    :param channels_last:
    :return:
    """

    # list data files, read timestamps from file
    velo_datetimes = read_timestamps_from_file(data_src['velodyne'].timestamps)
    image_datetimes = read_timestamps_from_file(data_src['image_02'].timestamps)
    filenames = get_files(data_src,
                          'velodyne', 'image_02', 'image_03', 'semantics_02', 'semantics_03',
                          'semantics_visual_02', 'semantics_visual_03',)

    # insert None for missing velodyne point clouds
    # insert None at the same positions for all velodyne datetime lists
    # Todo: naming mismatch 'velodyne_points' vs. 'velodyne'
    if missing_files['velodyne_points']:
        for i in missing_files['velodyne_points']:
            filenames['velodyne'] = np.insert(filenames['velodyne'], i, None)
            x = []
            for dt_list in velo_datetimes:
                x.append(np.insert(dt_list, i, None))
            velo_datetimes = tuple(x)

    it = iter(filenames.values())
    target_len = len(next(it))
    if not all(len(i) == target_len for i in it):
        raise ValueError("File lists do not have the same length.")

    target_len = len(filenames['velodyne'])
    if target_len != len(image_datetimes) != len(velo_datetimes[0]) != len(velo_datetimes[1]) != \
            len(velo_datetimes[2]):
        raise RuntimeError("Number of files and timestamps do not match.")

    if enable_ego_motion_correction:
        poses_interpolator, zero_time = prepare_ego_motion_interpolation(data_src)
    else:
        poses_interpolator, zero_time = None, None

    # output folder for ego-motion-corrected point clouds
    output_path_pointcloud = data_target / 'velodyne_points_corrected' / 'data'
    output_path_pointcloud.mkdir(parents=True, exist_ok=True)
    # output folder for semantic probabilities
    output_path_probs = data_target / 'lidar_semantics_deeplab_v3+71_90000'
    output_path_probs.mkdir(parents=False, exist_ok=True)
    # output folder for debug images
    output_path_debug = data_target / 'debug'
    output_path_debug.mkdir(parents=False, exist_ok=True)
    # output for sensor poses
    output_path_sensor_poses = data_target / 'poses_sensor'
    output_path_sensor_poses.mkdir(parents=False, exist_ok=True)
    # save stats and merge to per-sequence stats
    statistics = {'concensus_mean': np.empty(shape=(target_len,), dtype=np.float32),
                  'concensus_vote': np.empty(shape=(target_len,), dtype=np.float32), }

    # calc number of files
    samples_per_file = 100
    total_number_of_files = ((target_len - 1) // samples_per_file) + 1

    # filename formatting
    digits = math.ceil(math.log10(total_number_of_files + 1))
    f_str = '0{}d'.format(digits)
    filename_formatter = str(output_path_probs / 'probabilities_{{:{}}}_of_{{:{}}}.tfrecords'
                             .format(f_str, f_str))

    if progress_bar:
        bar = progressbar.ProgressBar(
            redirect_stdout=True,
            max_value=target_len)
    else:
        bar = None

    i = 0

    sensor_poses = []

    for file_counter in range(total_number_of_files):

        if i == target_len:
            break

        f_name = filename_formatter.format(
            file_counter, total_number_of_files)

        with tf.io.TFRecordWriter(f_name, options=tf.io.TFRecordCompressionType.GZIP) as writer:

            samples_written_in_file = 0
            while True:  # < loop over samples in current file

                if i not in missing_files['velodyne_points']:
                    point_cloud_corrected, probs, mapping, sensor_pose, stats, debug_imgs = \
                        process_single_example(
                            i,
                            filenames['velodyne'][i], filenames['image_02'][i],
                            filenames['image_03'][i],
                            filenames['semantics_02'][i],
                            filenames['semantics_03'][i],
                            filenames['semantics_visual_02'][i],
                            filenames['semantics_visual_03'][i],
                            scales, poses_interpolator, kitti, timestamps=(v[i]
                                                                           for v in velo_datetimes),
                            velo_calib=velo_calib, zero_time=zero_time,
                            enable_ego_motion_correction=enable_ego_motion_correction,
                            channels_last=channels_last,
                            last_example=i == target_len - 1
                        )
                    # write ego motion corrected point cloud
                    write_binary_point_cloud_with_intensity(str(output_path_pointcloud /
                                                                '{:010d}.bin'.format(i)),
                                                            point_cloud_corrected)

                    # save to write to disk in a single file later
                    sensor_poses.append(sensor_pose)

                    # write semantic predictions as tfrecord
                    writer.write(make_example_proto(probs, mapping).SerializeToString())
                    # save stats
                    statistics['concensus_mean'][i] = stats[2]
                    statistics['concensus_vote'][i] = stats[3]
                    # debug images
                    if debug_imgs:
                        debug_imgs[0].save(str(output_path_debug / 'debug_02_{:05d}.png'.format(i)))
                        debug_imgs[1].save(str(output_path_debug / 'debug_03_{:05d}.png'.format(i)))

                # looping
                if progressbar is not None:
                    bar.update(i)
                i += 1
                samples_written_in_file += 1
                if samples_written_in_file == samples_per_file or i == target_len:
                    break

    # save sensor poses (as .npy and .txt)
    sensor_poses = np.stack(sensor_poses, axis=0)
    np.save(str(output_path_sensor_poses / 'sensor_poses.npy'), sensor_poses)
    with open(str(output_path_sensor_poses / 'sensor_poses.txt'), 'w') as txt_file:
        for pose in sensor_poses:
            txt_file.write(' '.join(map(str, list(pose.flatten()))) + '\n')

    # save stats
    np.savez(str(output_path_probs / 'concensus_stats'), **statistics)
    logger.info("Sequence consensus [mean/vote]: {} / {}"
                .format(statistics['concensus_mean'].mean(), statistics['concensus_vote'].mean()))
