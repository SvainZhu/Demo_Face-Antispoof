#!/usr/bin/env python

import cv2, sys, os
import face_recognition
from math import ceil
from glob import glob

interpolation = cv2.INTER_CUBIC
borderMode = cv2.BORDER_REPLICATE


def crop_face(img, bbox, crop_sz, bbox_ext=0, extra_pad=0):
    shape = img.shape  # [height, width, channels]
    x, y, w, h = bbox

    jitt_pad = int(ceil(float(extra_pad) * min(w, h) / crop_sz))

    pad = 0
    if x < w * bbox_ext + jitt_pad:
        pad = max(pad, w * bbox_ext + jitt_pad - x)
    if x + w * (1 + bbox_ext) + jitt_pad > shape[1]:
        pad = max(pad, x + w * (1 + bbox_ext) + jitt_pad - shape[1])
    if y < h * bbox_ext + jitt_pad:
        pad = max(pad, h * bbox_ext + jitt_pad - y)
    if y + h * (1 + bbox_ext) + jitt_pad > shape[0]:
        pad = max(pad, y + h * (1 + bbox_ext) + jitt_pad - shape[0])
    pad = int(pad)

    if pad > 0:
        pad = pad + 3
        replicate = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderMode)
    else:
        replicate = img
    cropped = replicate[int(pad + y - h * bbox_ext - jitt_pad): int(pad + y + h * (1 + bbox_ext) + jitt_pad),
              int(pad + x - w * bbox_ext - jitt_pad): int(pad + x + w * (1 + bbox_ext) + jitt_pad)]
    resized = cv2.resize(cropped, (crop_sz + 2 * extra_pad, crop_sz + 2 * extra_pad), interpolation=interpolation)
    return resized


def bbox_txt(img):
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    if not face_locations:
        bbox = {}
        c = 'False'
        return bbox, c
    else:
        c = 'True'
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
         bbox = left, top, right - left, bottom - top
        return bbox, c


def process_db_casia(db_dir, save_dir, scale, crop_sz):
    for frame_name in glob('%s/*.jpg' % db_dir):

        frame_idx = frame_name.split('\\')[-1].split('.')[0]
        frame = cv2.imread(frame_name)
        bbox, c = bbox_txt(frame)
        if c == 'False':
            return 'False'

        else:
            bbox_ext = (scale - 1.0) / 2
            cropped = crop_face(frame, bbox, crop_sz, bbox_ext)
            save_fname = save_dir + '\\face_' + frame_idx + '.jpg'
            cv2.imwrite(save_fname, cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print('faces are cropped')
    return 'True'


def crop_images(data_dir, face_dir, scale, crop_sz):
    c = process_db_casia(data_dir, face_dir, scale, crop_sz)
    return c
