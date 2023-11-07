#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA


from copy import deepcopy
from random import randint

import cv2


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------
    train_image = cv2.imread("/home/mike/Desktop/GoogleDrive/UA/Aulas/2023-2024/1ºSem/SAVI_23-24/savi_23-24/Parte06/images/santorini/1.png")
    query_image = cv2.imread("/home/mike/Desktop/GoogleDrive/UA/Aulas/2023-2024/1ºSem/SAVI_23-24/savi_23-24/Parte06/images/santorini/2.png")


    height_t, width_t, nc_t = train_image.shape
    height_q, width_q, nc_q = query_image.shape
    # print(query_image.shape)
    # print(train_image.shape)

    # --------------------------------------
    # Execution
    # --------------------------------------

    # Sift features  -----------------------
    sift_detector = cv2.SIFT_create(nfeatures=500)

    t_key_points, t_descriptors = sift_detector.detectAndCompute(train_image, None)
    q_key_points, q_descriptors = sift_detector.detectAndCompute(query_image, None)

    # Draw the keypoints on the images
    train_image_gui = deepcopy(train_image)
    for key_point in t_key_points: # iterate all keypoints
        x, y = int(key_point.pt[0]), int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255),randint(0, 255))
        cv2.circle(train_image_gui, (x,y), 15, color, 1)

    query_image_gui = deepcopy(query_image)
    for key_point in q_key_points: # iterate all keypoints
        x, y = int(key_point.pt[0]), int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255),randint(0, 255))
        cv2.circle(query_image_gui, (x,y), 15, color, 1)

    # Visualization -----------------------
    cv2.namedWindow('train image', cv2.WINDOW_NORMAL)
    cv2.imshow('train image', train_image_gui)

    cv2.namedWindow('query image', cv2.WINDOW_NORMAL)
    cv2.imshow('query image', query_image_gui)

    cv2.waitKey(0)
    # --------------------------------------
    # Termination
    # --------------------------------------


if __name__ == "__main__":
    main()
