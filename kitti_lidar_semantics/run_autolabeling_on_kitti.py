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
import threading

import deeplab
from deeplab.vis import FLAGS as dl_flags
from deeplab.vis import main as dl_main

from .autolabeling_on_cartographer import do_work, Data
from .split_kitti_point_clouds import save_velo_data_stream
from .shared_snippets import read_xml_config
from .post_to_mattermost import send_mattermost_msg


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

KITTI_DAYS = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03', ]

DEBUG = True


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
    len_img2 = len(os.listdir(pathlib.Path(sequence[2]) / 'image_02' / 'data'))
    len_img3 = len(os.listdir(pathlib.Path(sequence[2]) / 'image_03' / 'data'))
    len_velo = len(os.listdir(pathlib.Path(sequence[2]) / 'velodyne_points' / 'data'))
    len_oxts = len(os.listdir(pathlib.Path(sequence[2]) / 'oxts' / 'data'))
    if not (len_img2 == len_img3 == len_velo == len_oxts):
        return None

    return len_img2


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
    sem_folder_name = 'sem_deeplab71_{}_{}'.format(checkpoint_iteration, input_suffix)
    output_path_tmp = tmp_dir / sem_folder_name
    output_path_tmp.mkdir(exist_ok=True, parents=True)
    output_path_tmp_data = output_path_tmp / 'data'
    output_path_sem = output_dir /\
                      'sem_deeplab71_{}_{}'.format(checkpoint_iteration, input_suffix)
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
    with (output_path_tmp / 'timestamps.txt').open(mode='xb') as tid:
        tid.write((input_path / 'timestamps.txt').read_bytes())
    with (output_path_sem / 'timestamps.txt').open(mode='xb') as tid:
        tid.write((input_path / 'timestamps.txt').read_bytes())

    return output_path_tmp, sem_folder_name


def check_paths(data):
    for v in data.values():
        if not v.data.exists():
            raise FileNotFoundError("Does not exist {}.".format(v.data))
        x = v.timestamps if isinstance(v.timestamps, tuple) else (v.timestamps, )
        for t in x:
            if not t.is_file():
                raise FileNotFoundError("Not a file {}.".format(t))


def process_sequence(checkpoint, kitti_root, cartographer_script, velo_calib, output_dir, sequence):
    count = get_raw_sequence_sample_count(sequence)
    if not count:
        err = "Invalid sequence {}_{}".format(sequence[0], sequence[1])
        return False, err, None

    # tmp_dir = tempfile.mkdtemp(prefix='kitti_semantics_{}_{}'.format(sequence[0], sequence[1]))
    # Todo DEBUG
    assert DEBUG
    tmp_dir = "/tmp/kitti_semantics_TMP"
    # shutil.rmtree(tmp_dir)
    # pathlib.Path(tmp_dir).mkdir(exist_ok=True)

    logger.info("Writing to tmp directory {}".format(tmp_dir))
    tmp_dir = pathlib.Path(tmp_dir)

    day_folder, seq_folder = pathlib.Path(sequence[2]).parts[-2:]
    processed_output = output_dir / day_folder / seq_folder
    processed_output.mkdir(parents=True, exist_ok=True)
    logger.info("Writing to output directory {}".format(str(processed_output)))

    log_data = {}

    try:
        # # RGB SEMANTICS
        # path_sem_02, sem_folder_02 = generate_stereo_semantics(
        #     tmp_dir, processed_output, checkpoint, sequence, count, input_suffix='02')
        # path_sem_03, sem_folder_03 = generate_stereo_semantics(
        #     tmp_dir, processed_output, checkpoint, sequence, count, input_suffix='03')
        #
        # # PARTITION POINT CLOUDS
        # # complete point cloud
        # path_partitioned = processed_output / 'velodyne_points_partitioned'
        # path_partitioned.mkdir(parents=True, exist_ok=True)
        # # point cloud without points z < -1.4m
        # path_partitioned2 = tmp_dir / 'velodyne_points_partitioned_truncated'
        # path_partitioned2.mkdir(parents=True, exist_ok=True)
        # save_velo_data_stream(velodyne_data_folder=pathlib.Path(sequence[2]) / 'velodyne_points',
        #                       velodyne_target_folder=str(path_partitioned),
        #                       velodyne_target_folder2=str(path_partitioned2),
        #                       velo_calib=velo_calib)
        #
        # # CARTOGRAPHER
        # print("Running cartographer")
        # if subprocess.call([str(cartographer_script), str(kitti_root), str(tmp_dir), sequence[0],
        #                     sequence[1], str(processed_output)]):
        #     raise RuntimeError("Error when calling cartographer.")

        # # Todo
        # assert DEBUG
        path_sem_02 = pathlib.Path('/tmp/kitti_semantics_TMP/sem_deeplab71_90000_02')
        path_sem_03 = pathlib.Path('/tmp/kitti_semantics_TMP/sem_deeplab71_90000_03')
        sem_folder_02 = 'sem_deeplab71_90000_02'
        sem_folder_03 = 'sem_deeplab71_90000_03'

        # EGO MOTION CORRECTION + AUTOLABELING
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
                channels_last=True)

        bkp = pathlib.Path('/mapr/738.mbc.de/data/transform/rd/athena/atplid/'
                           'processed_raw_sequences') / day_folder
        x = threading.Thread(target=copy_to_mapr, args=(processed_output, bkp, bkp / seq_folder))
        # # Todo Debug
        # assert DEBUG
        # x.start()
        x = None

        # create a success marker
        with open(str(processed_output / 'log'), 'w') as f:
            f.write('complete ' + str(datetime.datetime.now()))

    except Exception as e:
        print("Encountered an error. Skipping. Message: {}".format(str(e)))
        # create a fail marker
        with open(str(processed_output / 'log'), 'w') as f:
            f.write('failed ' + str(datetime.datetime.now()) + str(e))
        return False, str(e), None
    finally:
        # Todo DEBUG
        assert DEBUG
        # shutil.rmtree(tmp_dir)
    return True, 'ok', x


def copy_to_mapr(src, dst_day: pathlib.PosixPath, dst_seq: pathlib.PosixPath):
    dst_day.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(src), str(dst_seq))


@click.command()
@click.argument('kitti_root')
@click.argument('output')
@click.argument('checkpoint')
def main(kitti_root, output, checkpoint):

    import py_riches
    cartographer_script = (pathlib.Path(py_riches.__file__) / '..'
                           / 'script' / 'run_cartographer_on_sequence.sh').resolve()
    if not cartographer_script.exists():
        raise RuntimeError("Cartographer script not found {}".format(cartographer_script))

    calib_file = pathlib.Path(__file__).resolve().parent / pathlib.Path('data') / \
                 pathlib.Path('hdl64_calib_corrections.xml')
    velo_calib = read_xml_config(calib_file)

    kitti_root = pathlib.Path(kitti_root)
    output = pathlib.Path(output)
    sequences = find_kitti_raw_sequences(kitti_root, KITTI_DAYS)

    threads = []

    for s in sequences:
        # Todo DEBUG
        assert DEBUG
        # s = sequences[22]
        # s = sequences[65]
        # s = sequences[141]
        s = sequences[24]

        logger.info("Processing Sequence {}/{}".format(s[0], s[1]))
        succ, msg, thread = process_sequence(checkpoint, kitti_root, cartographer_script,
                                             velo_calib, output, s)
        if thread:
            threads.append(thread)
        if succ:
            msg = '{} [AUTOLABELING] Processed KITTI sequence {}/{} successfully.'\
                .format(str(datetime.datetime.now()), s[0], s[1])
        else:
            msg = '{} [AUTOLABELING] Failed on KITTI sequence {}/{}: {}.'\
                .format(str(datetime.datetime.now()), s[0], s[1], msg)
        try:
            if not send_mattermost_msg(msg):
                print("Could not post to mattermost. Msg: {}".format(msg))
        except Exception as e:
            print("Could not send to mattermost {}".format(str(e)))

        # # Todo DEBUG
        assert DEBUG
        break

    print("Waiting for copy threads")
    for t in threads:
        try:
            t.join()
        except RuntimeError():
            pass


if __name__ == '__main__':
    main()
