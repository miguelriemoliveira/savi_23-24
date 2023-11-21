#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA


from copy import deepcopy
from random import randint

import cv2
import numpy as np


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------
    train_image = cv2.imread(
        "/home/mike/Desktop/GoogleDrive/UA/Aulas/2023-2024/1ºSem/SAVI_23-24/savi_23-24/Parte06/images/machu_pichu2/1.png")
    query_image = cv2.imread(
        "/home/mike/Desktop/GoogleDrive/UA/Aulas/2023-2024/1ºSem/SAVI_23-24/savi_23-24/Parte06/images/machu_pichu2/2.png")

    train_image_gui = deepcopy(train_image)
    query_image_gui = deepcopy(query_image)

    # --------------------------------------
    # Execution
    # --------------------------------------

    # Sift features  -----------------------
    sift_detector = cv2.SIFT_create(nfeatures=100)

    t_key_points, t_descriptors = sift_detector.detectAndCompute(train_image, None)
    q_key_points, q_descriptors = sift_detector.detectAndCompute(query_image, None)

    # Match the keypoints
    index_params = dict(algorithm=1, trees=15)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    two_best_matches = flann_matcher.knnMatch(q_descriptors, t_descriptors, k=2)

    # Create a list of matches
    matches = []
    for match_idx, match in enumerate(two_best_matches):

        best_match = match[0]  # to get the cv2.DMatch from the tuple [match = (cv2.DMatch)]
        second_match = match[1]

        # David Lowe's ratio
        if best_match.distance < 0.8 * second_match.distance:  # this is a robust match, keep it
            matches.append(best_match)  # create a list to show with drawMatches

    matches_image = cv2.drawMatches(query_image, q_key_points, train_image, t_key_points, matches, None)

    # Compute the transformation between images (homography)
    # q_key_points_nparray should be a numpy ndarray with size (n, 1, 2) of type np.float32

    # Initialize the numpy arrays
    n = len(matches)
    q_key_points_nparray = np.ndarray((n, 1, 2), dtype=np.float32)
    t_key_points_nparray = np.ndarray((n, 1, 2), dtype=np.float32)

    # Set the proper values
    for match_idx, match in enumerate(matches):
        q_idx = match.queryIdx
        t_idx = match.trainIdx

        x_q, y_q = q_key_points[q_idx].pt[0], q_key_points[q_idx].pt[1]
        x_t, y_t = t_key_points[t_idx].pt[0], t_key_points[t_idx].pt[1]

        t_key_points_nparray[match_idx, 0, 0] = x_t
        t_key_points_nparray[match_idx, 0, 1] = y_t

        q_key_points_nparray[match_idx, 0, 0] = x_q
        q_key_points_nparray[match_idx, 0, 1] = y_q

    H, _ = cv2.findHomography(q_key_points_nparray, t_key_points_nparray, cv2.RANSAC)

    height_t, width_t, _ = train_image.shape
    height_q, width_q, _ = query_image.shape
    print('height_t=' + str(height_t) + ' width_t=' + str(width_t))

    # transform top left corner
    corners = np.array([[0, 0], [width_q-1, 0],
                        [width_q-1, height_q-1],
                        [0, height_q-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Compute the size of the stitched image
    corners_transformed = cv2.perspectiveTransform(corners, H)
    corners_transformed = corners_transformed.tolist()

    # compute min and max using q_transformed corners
    min_x_q, max_x_q, min_y_q, max_y_q = None, None, None, None
    for corner in corners_transformed:
        x = corner[0][0]
        y = corner[0][1]

        if min_x_q is None or x < min_x_q:
            min_x_q = int(round(x))

        if min_y_q is None or y < min_y_q:
            min_y_q = int(round(y))

        if max_x_q is None or x > max_x_q:
            max_x_q = int(round(x))

        if max_y_q is None or y > max_y_q:
            max_y_q = int(round(y))

    print('min_x = ' + str(min_x_q))
    print('max_x = ' + str(max_x_q))

    print('min_y = ' + str(min_y_q))
    print('max_y = ' + str(max_y_q))

    stitch_image_w = max(max_x_q, width_t) - min(min_x_q, 0)
    stitch_image_h = max(max_y_q, height_t) - min(min_y_q, 0)

    print('stitch_w = ' + str(stitch_image_w))
    print('stitch_h = ' + str(stitch_image_h))

    query_image_transformed = cv2.warpPerspective(query_image, H,
                                                  (stitch_image_w, stitch_image_h))

    # How to transform the target image
    tx = -min(min_x_q, 0)
    ty = -min(min_y_q, 0)

    print('H=\n' + str(H))
    H_t = np.eye(3, dtype=np.float32)
    H_t[0, 2] = tx
    H_t[1, 2] = 200

    print('H_t=\n' + str(H_t))

    train_image_transformed = cv2.warpPerspective(train_image, H_t,
                                                  (stitch_image_w, stitch_image_h))

    # Stitching --------------------------------------------
    # basic stitching, merge all pixels
    stitched_image = cv2.addWeighted(train_image_transformed, 0.5, query_image_transformed, 0.5, gamma=0)

    # Advanced stitching, use a mask of used pixels in the query_image_transformed
#     query_image_transformed_gray = cv2.cvtColor(query_image_transformed, cv2.COLOR_BGR2GRAY)
#     mask_used_pixels = query_image_transformed_gray > 0
#     print(mask_used_pixels.dtype)
#
#     stitched_image = train_image_transformed.astype(float)
#     stitched_image[mask_used_pixels] = train_image_transformed[mask_used_pixels].astype(float) * 0.5 + \
#         query_image_transformed[mask_used_pixels].astype(float) * 0.5
#
#     stitched_image = stitched_image.astype(np.uint8)

    # Visualization -----------------------

    # Draw the keypoints on the images
    for key_point in t_key_points:  # iterate all keypoints
        x, y = int(key_point.pt[0]), int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(train_image_gui, (x, y), 15, color, 1)

    for key_point in q_key_points:  # iterate all keypoints
        x, y = int(key_point.pt[0]), int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(query_image_gui, (x, y), 15, color, 1)

    cv2.namedWindow('train image', cv2.WINDOW_NORMAL)
    cv2.imshow('train image', train_image_gui)

    # cv2.namedWindow('query image', cv2.WINDOW_NORMAL)
    # cv2.imshow('query image', query_image_gui)

    cv2.namedWindow('query image transformed', cv2.WINDOW_NORMAL)
    cv2.imshow('query image transformed', query_image_transformed)
    cv2.namedWindow('train image transformed', cv2.WINDOW_NORMAL)
    cv2.imshow('train image transformed', train_image_transformed)

    # cv2.namedWindow('mask_used_pixels', cv2.WINDOW_NORMAL)
    # cv2.imshow('mask_used_pixels', mask_used_pixels.astype(np.uint8)*255)

    # cv2.namedWindow('matches image', cv2.WINDOW_NORMAL)
    # cv2.imshow('matches image', matches_image)

    cv2.namedWindow('stitched image', cv2.WINDOW_NORMAL)
    cv2.imshow('stitched image', stitched_image)

    cv2.waitKey(0)
    # --------------------------------------
    # Termination
    # --------------------------------------


if __name__ == "__main__":
    main()
