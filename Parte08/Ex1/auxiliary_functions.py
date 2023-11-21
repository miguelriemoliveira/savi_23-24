#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA


from copy import deepcopy
import copy
from random import randint

import cv2
import numpy as np


def showFloatImage(win_name, image):
    image_uint8 = image.astype(np.uint8)
    cv2.imshow(win_name, image_uint8)


def computeOverlapMask(q_image):
    q_image_gray = cv2.cvtColor(q_image, cv2.COLOR_BGR2GRAY)
    mask_used_pixels = q_image_gray > 0

    return mask_used_pixels


def stitchImages(t_image, q_image, mask):
    # basic stitching, merge all pixels
    # stitched_image = cv2.addWeighted(train_image,0.5, query_image_transformed, 0.5, gamma=0)

    if q_image.dtype == np.uint8:  # must convert
        q_image = q_image.astype(float)

    if t_image.dtype == np.uint8:  # must convert
        t_image = t_image.astype(float)

    # Advanced stitching, use a mask of used pixels in the query_image_transformed
    stitched_image = deepcopy(t_image)
    stitched_image[mask] = t_image[mask].astype(float) * 0.5 + \
        q_image[mask].astype(float) * 0.5

    return stitched_image


def objectiveFunction(params, data):

    # -----------------------------------
    # extract parameters from vector
    # -----------------------------------
    s_q = params[0]
    b_q = params[1]
    s_t = params[2]
    b_t = params[3]

    # -----------------------------------
    # apply the model (obtain the corrected images)
    # -----------------------------------
    q_image_corrected = data['q_image_original'] * s_q + b_q
    # TODO should we do this?
    # TODO perhaps touch only used pixels
    q_image_corrected = np.minimum(q_image_corrected, 255)
    q_image_corrected = np.maximum(q_image_corrected, 0)

    t_image_corrected = data['t_image_original'] * s_t + b_t
    t_image_corrected = np.minimum(t_image_corrected, 255)
    t_image_corrected = np.maximum(t_image_corrected, 0)

    # -----------------------------------
    # compute error
    # -----------------------------------
    similarity_error = np.average(np.abs(q_image_corrected[data['overlap_mask']] -
                                         t_image_corrected[data['overlap_mask']]))
    print('similarity error = ' + str(similarity_error))

    # Compute error related to absolute color
    avg_q = np.average(q_image_corrected[data['overlap_mask']])
    avg_t = np.average(t_image_corrected[data['overlap_mask']])
    target_brightness = 170

    reference_error_q = target_brightness - avg_q
    print('reference_error_q = ' + str(reference_error_q))
    reference_error_t = target_brightness - avg_t
    print('reference_error_t = ' + str(reference_error_t))

    # -----------------------------------
    # Visualization
    # -----------------------------------

    # run stitching to show stitched image again
    stitched_image = stitchImages(t_image_corrected, q_image_corrected, data['overlap_mask'])

    showFloatImage('t_image', t_image_corrected)
    showFloatImage('q_image', q_image_corrected)
    showFloatImage('stitched image', stitched_image)
    cv2.waitKey(25)

    return [similarity_error, reference_error_q, reference_error_t]
