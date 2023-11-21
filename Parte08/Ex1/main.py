#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA


from copy import deepcopy
from functools import partial
from random import randint

import cv2
import numpy as np
from auxiliary_functions import showFloatImage, stitchImages, objectiveFunction, computeOverlapMask

from scipy.optimize import least_squares


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------
    t_image = cv2.imread(
        "/home/mike/Desktop/GoogleDrive/UA/Aulas/2023-2024/1ºSem/SAVI_23-24/savi_23-24/Parte06/images/machu_pichu/1.png")
    q_image = cv2.imread("./query_image_transformed.png")

    mask = computeOverlapMask(q_image)

    stitched_image = stitchImages(t_image, q_image, mask)

    # Visualize initial images
    cv2.namedWindow('t_image', cv2.WINDOW_NORMAL)
    cv2.imshow('t_image', t_image)

    cv2.namedWindow('q_image', cv2.WINDOW_NORMAL)
    cv2.imshow('q_image', q_image)

    cv2.namedWindow('stitched image', cv2.WINDOW_NORMAL)
    showFloatImage('stitched image', stitched_image)

    # Define data to use in objective function
    data = {'q_image_original': q_image.astype(float),
            't_image_original': t_image.astype(float),
            'overlap_mask': mask}

    # Define parameters to use in optimization
    #         s_q    b_q     s_t     b_t
    # params = [0.5,    -5,     2.0,     0]
    params0 = [1.0,    0,     1.0,     0]

    # --------------------------------------
    # Execution
    # --------------------------------------

    # To test, call objective function only once
    # objectiveFunction(params0, data)

    result = least_squares(partial(objectiveFunction, data=data), params0, verbose=2)

    print('OPTIMIZATION FINISHED')
    print(result)
    cv2.waitKey(0)

    # --------------------------------------
    # Termination
    # --------------------------------------


if __name__ == "__main__":
    main()
