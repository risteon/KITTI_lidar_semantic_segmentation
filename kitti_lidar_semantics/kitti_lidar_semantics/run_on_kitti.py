#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
import pathlib
import click
import shutil
import logging
import glob
import pykitti
import itertools
import tempfile
import subprocess
import datetime
import time
import sys
import traceback
import threading

import deeplab
from deeplab.vis import FLAGS as dl_flags
from deeplab.vis import main as dl_main

from .ego_motion_correction_and_interpolation import do_work, Data
from .utils import save_velo_data_stream
from .utils import read_xml_config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

KITTI_DAYS = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03', ]


def find_kitti_raw_sequences(kitti_root, kitti_days):
    for day in kitti_days:
        if not (kitti_root / pathlib.Path(day)).is_dir():
            raise RuntimeError("Invalid KITTI Root")

    sequences = []
    for day in kitti_days:
        sequence_paths = sorted(glob.glob(
            str(kitti_root / pathlib.Path(day) / pathlib.Path('{}_drive_*_sync'.format(day)))))
        sequence_numbers = [os.path.basename(x)[-9:-5] for x in sequence_paths]
        sequences.extend(list(zip(itertools.repeat(day), sequence_numbers, sequence_paths)))
    return sequences


def get_raw_sequence_sample_count(sequence):
    sequence_root = pathlib.Path(sequence[2])
    data = ['image_02', 'image_03', 'velodyne_points', 'oxts']
    files = [sorted(list((sequence_root / x / 'data').iterdir())) for x in data]
    lenghts = [len(x) for x in files]
    stems = [[y.stem for y in x] for x in files]

    missing = {k: set() for k in data}
    for i in range(max(lenghts)):
        for x, src in zip(stems, data):
            if '{:010d}'.format(i) not in x:
                logger.warning("Sample #{} missing in {}".format(i, src))
                missing[src].add(i)
    return max(lenghts), missing


def make_rgb_semantics(input_folder, output_folder, checkpoint):
    output_folder.mkdir(exist_ok=True, parents=False)

    dl_flags.model_variant = 'xception_71'
    dl_flags.dense_prediction_cell_json = os.path.abspath(os.path.join(
        deeplab.core.__file__, '..', 'dense_prediction_cell_branch5_top1_cityscapes.json'))
    dl_flags.decoder_output_stride = [4]
    dl_flags.vis_batch_size = 1
    dl_flags.dataset_dir = str(input_folder)
    dl_flags.vis_split = 'training'
    dl_flags.colormap_type = 'cityscapes'
    dl_flags.vis_crop_size = [385, 1249]
    dl_flags.folder_png_mode = True
    dl_flags.eval_scales = [0.25, 0.75, 2.0]
    dl_flags.add_flipped_images = True
    dl_flags.prob_output_mode = True

    dl_flags.save_original_image = False
    dl_flags.vis_logdir = str(output_folder)
    dl_flags.checkpoint_dir = checkpoint
    dl_flags.dataset = 'kitti'
    # run
    dl_main(None)


def generate_stereo_semantics(tmp_dir, output_dir, checkpoint, sequence, expected_count,
                              input_suffix):
    checkpoint_iteration = pathlib.Path(checkpoint).name.split('-')[-1]
    sem_folder_name = 'sem_deeplab_v3+71_{}_{}'.format(checkpoint_iteration, input_suffix)
    output_path_tmp = tmp_dir / sem_folder_name
    output_path_tmp.mkdir(exist_ok=True, parents=True)
    output_path_tmp_data = output_path_tmp / 'data'
    output_path_sem = output_dir /\
                      'sem_deeplab_v3+71_{}_{}'.format(checkpoint_iteration, input_suffix)
    output_path_sem_data = output_path_sem / 'data'

    input_path = pathlib.Path(sequence[2]) / 'image_{}'.format(input_suffix)
    make_rgb_semantics(input_folder=input_path / 'data',
                       output_folder=output_path_tmp_data,
                       checkpoint=checkpoint)

    # copy generated pngs to processed output folder
    sem_pngs = [x for x in output_path_tmp_data.iterdir() if x.suffix == '.png']
    if len(sem_pngs) != expected_count:
        raise RuntimeError(
            "Invalid number of generated semantic png's in {}.".format(output_path_tmp_data))

    # copy RGB semantics to processed output folder
    if output_path_sem_data.exists():
        shutil.rmtree(output_path_sem_data)
    output_path_sem_data.mkdir(exist_ok=False, parents=True)
    targets = [output_path_sem_data / x.name for x in sem_pngs]
    for png, target in zip(sem_pngs, targets):
        with target.open(mode='xb') as tid:
            tid.write(png.read_bytes())

    # copy timestamps
    try:
        with (output_path_tmp / 'timestamps.txt').open(mode='xb') as tid:
            tid.write((input_path / 'timestamps.txt').read_bytes())
    except FileExistsError:
        pass
    try:
        with (output_path_sem / 'timestamps.txt').open(mode='xb') as tid:
            tid.write((input_path / 'timestamps.txt').read_bytes())
    except FileExistsError:
        pass

    return sem_folder_name


def check_paths(data):
    for v in data.values():
        if not v.data.exists():
            raise FileNotFoundError("Does not exist {}.".format(v.data))
        x = v.timestamps if isinstance(v.timestamps, tuple) else (v.timestamps, )
        for t in x:
            if not t.is_file():
                raise FileNotFoundError("Not a file {}.".format(t))


def process_sequence_part_a(checkpoint, kitti_root, cartographer_script, velo_calib, output_dir,
                            sequence):
    logger.info("Processing Sequence {}/{} PART A".format(sequence[0], sequence[1]))
    count, missing = get_raw_sequence_sample_count(sequence)
    # in sequence 2011-09-26/0009 4 velodyne frames are missing.
    if any(v for v in missing.values()) and len(missing['velodyne_points']) > 5:
        err = "Invalid sequence {}_{}".format(sequence[0], sequence[1])
        return False, err

    tmp_dir = tempfile.mkdtemp(prefix='kitti_semantics_{}_{}'.format(sequence[0], sequence[1]))
    logger.info("Writing to tmp directory {}".format(tmp_dir))
    tmp_dir = pathlib.Path(tmp_dir)

    day_folder, seq_folder = pathlib.Path(sequence[2]).parts[-2:]
    processed_output = output_dir / day_folder / seq_folder
    processed_output.mkdir(parents=True, exist_ok=True)
    logger.info("Writing to output directory {}".format(str(processed_output)))

    try:
        # RGB SEMANTICS
        logger.info("Creating RGB semantics for left image 02.")
        sem_folder_02 = generate_stereo_semantics(
            tmp_dir, processed_output, checkpoint, sequence, count, input_suffix='02')
        logger.info("Creating RGB semantics for right image 03.")
        sem_folder_03 = generate_stereo_semantics(
            tmp_dir, processed_output, checkpoint, sequence, count, input_suffix='03')

    except Exception as e:
        tb = traceback.format_exc()
        print("Encountered an error. Skipping. Message: {}".format(str(e)))
        # create a fail marker
        with open(str(processed_output / 'log'), 'w') as f:
            f.write('failed ' + str(datetime.datetime.now()) + ': ' + str(e))
            f.write('\n{}'.format(tb))
        # delete tmp in case of error. Otherwise will be removed when part_b has finished
        shutil.rmtree(tmp_dir)
        return False, str(e)

    data = [tmp_dir, sem_folder_02, sem_folder_03, missing]
    return True, data


def process_sequence_part_b(data_from_part_a, kitti_root, cartographer_script, velo_calib,
                            output_dir, sequence, results):
    logger.info("Processing Sequence {}/{} PART B".format(sequence[0], sequence[1]))
    tmp_dir, sem_folder_02, sem_folder_03, missing = data_from_part_a
    path_sem_02 = tmp_dir / sem_folder_02
    path_sem_03 = tmp_dir / sem_folder_03

    day_folder, seq_folder = pathlib.Path(sequence[2]).parts[-2:]
    processed_output = output_dir / day_folder / seq_folder

    try:
        # PARTITION POINT CLOUDS
        logger.info("Partitioning point clouds.")
        # complete point cloud
        path_partitioned = processed_output / 'velodyne_points_partitioned'
        path_partitioned.mkdir(parents=True, exist_ok=True)
        # point cloud without points z < -1.4m
        path_partitioned2 = tmp_dir / 'velodyne_points_partitioned_truncated'
        path_partitioned2.mkdir(parents=True, exist_ok=True)
        save_velo_data_stream(velodyne_data_folder=pathlib.Path(sequence[2]) / 'velodyne_points',
                              velodyne_target_folder=str(path_partitioned),
                              velodyne_target_folder2=str(path_partitioned2),
                              velo_calib=velo_calib, missing_files=missing['velodyne_points'])

        # CARTOGRAPHER
        logger.info("Running cartographer.")
        # redirect output ...
        stdout = open(str(tmp_dir / "stdout.txt"), "wb")
        stderr = open(str(tmp_dir / "stderr.txt"), "wb")
        if subprocess.call([str(cartographer_script), str(kitti_root), str(tmp_dir), sequence[0],
                            sequence[1], str(processed_output)], stdout=stdout, stderr=stderr):
            raise RuntimeError("Error when calling cartographer.")

        # EGO MOTION CORRECTION + INTERPOLATION OF PROBS
        pykitti_obj = pykitti.raw(kitti_root, sequence[0], sequence[1])
        s_root = pathlib.Path(sequence[2])

        data = {
            'velodyne': Data(timestamps=(s_root / 'velodyne_points/timestamps.txt',
                                         s_root / 'velodyne_points/timestamps_start.txt',
                                         s_root / 'velodyne_points/timestamps_end.txt',),
                             data=s_root / 'velodyne_points/data',
                             file_extension='bin'),
            'map_cartographer': Data(
                timestamps=processed_output / 'poses_cartographer/timestamps.txt',
                data=processed_output / 'poses_cartographer/poses.txt',
                file_extension=None
            ),
            'image_02': Data(timestamps=s_root / 'image_02/timestamps.txt',
                             data=s_root / 'image_02/data',
                             file_extension='png'),
            'image_03': Data(timestamps=s_root / 'image_03/timestamps.txt',
                             data=s_root / 'image_03/data',
                             file_extension='png'),
            'semantics_02': Data(timestamps=path_sem_02 / 'timestamps.txt',
                                 data=path_sem_02 / 'data',
                                 file_extension='npz'),
            'semantics_03': Data(timestamps=path_sem_03 / 'timestamps.txt',
                                 data=path_sem_03 / 'data',
                                 file_extension='npz'),
            'semantics_visual_02': Data(
                timestamps=processed_output / sem_folder_02 / 'timestamps.txt',
                data=processed_output / sem_folder_02 / 'data',
                file_extension='png'
            ),
            'semantics_visual_03': Data(
                timestamps=processed_output / sem_folder_03 / 'timestamps.txt',
                data=processed_output / sem_folder_03 / 'data',
                file_extension='png'
            ),
        }

        check_paths(data)
        logger.info("Starting Ego-Motion correction and interpolation")
        do_work(pykitti_obj, data_src=data, data_target=processed_output, velo_calib=velo_calib,
                scales=['0.25_a', '0.25_b', '0.75_a', '0.75_b', '2.0_a', '2.0_b'],
                channels_last=True,
                missing_files=missing)

        # create a success marker
        with open(str(processed_output / 'log'), 'w') as f:
            f.write('complete ' + str(datetime.datetime.now()))

    except Exception as e:
        tb = traceback.format_exc()
        print("Encountered an error. Skipping. Message: {}".format(str(e)))
        # create a fail marker
        with open(str(processed_output / 'log'), 'w') as f:
            f.write('failed ' + str(datetime.datetime.now()) + ': ' + str(e))
            f.write('\n{}'.format(tb))

        results.extend([False, str(e), sequence])
        return
    finally:
        # Todo DEBUG
        pass
        # shutil.rmtree(tmp_dir)
    results.extend([True, 'ok', sequence])
    return


@click.command()
@click.argument('kitti_root')
@click.argument('output')
@click.argument('checkpoint')
@click.option('--day', default=None)
@click.option('--start-at', default=None)
def main(kitti_root, output, checkpoint, day, start_at):

    cartographer_script = (pathlib.Path(__file__) / '..' / '..' /
                           'script' / 'run_cartographer_on_sequence.sh').resolve()
    if not cartographer_script.exists():
        raise RuntimeError("Cartographer script not found {}".format(cartographer_script))

    calib_file = pathlib.Path(__file__).resolve().parent.parent / pathlib.Path('data') / \
                 pathlib.Path('hdl64_calib_corrections.xml')
    velo_calib = read_xml_config(calib_file)

    if day is None:
        kitti_days = KITTI_DAYS
    else:
        kitti_days = [day]
        del sys.argv[1:]

    kitti_root = pathlib.Path(kitti_root)
    output = pathlib.Path(output)
    sequences = find_kitti_raw_sequences(kitti_root, kitti_days)
    if start_at is not None:
        sequences = [x for x in sequences if x[1] >= start_at]

    thread_ego_motion = None
    results = []

    def wait_for_part_b(thread_ego_motion, log_file, results):
        if thread_ego_motion is not None:
            thread_ego_motion.join()
            assert len(results) == 3
            succ_part_b, msg_part_b, b_sequence = results
            if succ_part_b:
                msg_part_b = '{} Processed KITTI sequence {}/{} PART B successfully.' \
                    .format(str(datetime.datetime.now()), b_sequence[0], b_sequence[1])
            else:
                msg_part_b = '{} Failed on KITTI sequence {}/{} PART B: {}.' \
                    .format(str(datetime.datetime.now()), b_sequence[0], b_sequence[1], msg_part_b)
            try:
                log_file.write('{}\n'.format(msg_part_b))
            except Exception as e:
                print("Could not write log: {}".format(str(e)))
            results.clear()

    with open(str(output / 'log_{}'.format(time.strftime("%Y%m%d-%H%M%S"))), 'w') as log_file:
        for s in sequences:

            succ, msg = process_sequence_part_a(checkpoint, kitti_root,
                                                cartographer_script, velo_calib,
                                                output, s)
            if succ:
                data_part_a = msg
                msg = '{} Processed KITTI sequence {}/{} PART A successfully.'\
                    .format(str(datetime.datetime.now()), s[0], s[1])

                # maximum of one parallel part b thread. Wait if not already finished.
                wait_for_part_b(thread_ego_motion, log_file, results)
                thread_ego_motion = threading.Thread(
                    target=process_sequence_part_b,
                    name='thread_{}_{}'.format(s[0], s[1]),
                    args=(data_part_a, kitti_root, cartographer_script, velo_calib, output, s,
                          results))
                thread_ego_motion.start()

            else:
                msg = '{} Failed on KITTI sequence {}/{} PART A: {}.'\
                    .format(str(datetime.datetime.now()), s[0], s[1], msg)
            try:
                logger.info(msg)
                log_file.write('{}\n'.format(msg))
                log_file.flush()
            except Exception as e:
                print("Could not write log: {}".format(str(e)))

        wait_for_part_b(thread_ego_motion, log_file, results)


if __name__ == '__main__':
    main()
