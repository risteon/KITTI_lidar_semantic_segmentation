#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful code snippets. Be careful with non-standard module imports!
"""
# only standard includes!
import datetime
import operator
import pathlib
import struct
import imghdr
import numpy as np
import scipy.stats


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
    from imageio import imread
    return np.array(imread(png_image_path))


def get_image_size(image_path):
    """Determine the image type of fhandle and return its size.
    """
    from PIL import Image
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


def jsd(p: np.ndarray, q: np.ndarray, base=np.e):
    """
        Implementation of pairwise Jensen-Shannon Divergence based on
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    """
    # normalize p, q to probabilities
    p, q = p / p.sum(), q / q.sum()
    m = 1. / 2 * (p + q)
    return scipy.stats.entropy(p, m, base=base) / 2. + scipy.stats.entropy(q, m, base=base) / 2.


def resize_image_with_crop_or_pad(img, target_dims):
    if img.shape[:len(target_dims)] != target_dims:
        raise NotImplementedError()
    return img


def make_mosaic(data, nb_rows=None, nb_cols=None, px_margin=0, overlay_text=None):
    """
    If data is a list of non-equal sized images the first image determines the mosaic's size.
    """
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 12)

    if len(data[0].shape) != 3:
        raise RuntimeError("invalid data input format")

    # Calculate number of rows and cols in mosaic
    nb_data = len(data)
    if nb_cols is None or nb_rows is None:
        if nb_cols is None and nb_rows is None:
            nb_cols = 1
        if nb_cols is None:
            nb_cols = (nb_data + nb_rows - 1) // nb_rows
        elif nb_rows is None:
            nb_rows = (nb_data + nb_cols - 1) // nb_cols
    # Create empty target array with correct dimensions
    img_shape = data[0].shape[0:2]
    img_height = img_shape[0] + px_margin
    img_width = img_shape[1] + px_margin
    canvas_px_height = nb_rows * img_height + px_margin
    canvas_px_width = nb_cols * img_width + px_margin

    shape = (canvas_px_height, canvas_px_width, data[0].shape[2])

    mosaic = np.zeros(shape, dtype=data[0].dtype)
    # paste all images into mosaic
    for i in range(len(data)):
        r = i // nb_cols
        c = i % nb_cols
        mosaic[(r * img_height + px_margin):(r + 1) * img_height,
               (c * img_width + px_margin):(c + 1) * img_width, ...] = \
            resize_image_with_crop_or_pad(data[r * nb_cols + c], img_shape)

    img = Image.fromarray(mosaic)

    if overlay_text:
        assert len(overlay_text) == len(data)
        draw = ImageDraw.Draw(img)

        for i in range(len(data)):
            if overlay_text[i] is None:
                continue
            r = i // nb_cols
            c = i % nb_cols
            draw.text(((c * img_width + px_margin), (r * img_height + px_margin)),
                      overlay_text[i], (255, 255, 255), font=font)

    return img
