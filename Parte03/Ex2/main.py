#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA

import copy
import time

import cv2
import numpy as np


def main():

    roi0 = {'point_start': (404, 276), 'point_end': (460, 422), 'average':130,
        'average_previous':130, 'tic':time.time(), 'num_cars': 0}
    roi1 = {'point_start': (536, 285), 'point_end': (639, 414), 'average':130,
        'average_previous':130, 'tic':time.time(), 'num_cars': 0}
    roi2 = {'point_start': (687, 362), 'point_end': (847, 515), 'average':130,
        'average_previous':130, 'tic':time.time(), 'num_cars': 0}
    roi3 = {'point_start': (799, 219), 'point_end': (934, 318), 'average':130,
        'average_previous':130, 'tic':time.time(), 'num_cars': 0}

    rois = [roi0, roi1, roi2, roi3]
    print('List of rois = ' + str(rois) + '\n\n')
    
    # Load the image
    cap = cv2.VideoCapture('../docs/traffic.mp4')

    frame_number = 0
    num_cars = 0
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, image_rgb = cap.read()
        if ret is False:
            break

        image_gui = copy.deepcopy(image_rgb)

        # Convert to gray
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        for roi in rois:
            # Get sub_image
            point_start = roi['point_start']
            point_end = roi['point_end']

            image_roi = image_gray[point_start[1]:point_end[1] , point_start[0]:point_end[0]]

            # Compute average
            roi['average_previous'] = roi['average']
            roi['average'] = np.mean(image_roi)
            
            # Blackout
            t = 10.0
            blackout_threshold = 1.0
            time_since_tic = time.time() - roi['tic']

            if abs(roi['average'] - roi['average_previous']) > t and time_since_tic > blackout_threshold:
                roi['tic'] = time.time()
                roi['num_cars'] = roi['num_cars'] + 1 # Assume a change as a new car


        # --------------------------------------
        # Visualization
        # --------------------------------------


        image_gui = cv2.putText(image_gui, 'Frame ' + str(frame_number), (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255,255,0), 2, cv2.LINE_AA)


        for roi in rois:
            point_start = roi['point_start']
            point_end = roi['point_end']
            cv2.rectangle(image_gui, (point_start[0], point_start[1]), (point_end[0], point_end[1]), (0,255,0), 4)

            image_gui = cv2.putText(image_gui, 'Avg ' + str(round(roi['average'],1)), (point_start[0], 45), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0,255,255), 2, cv2.LINE_AA)
            image_gui = cv2.putText(image_gui, 'NCars ' + str(round(roi['num_cars'],1)), (point_start[0], 70), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0,255,0), 2, cv2.LINE_AA)


        cv2.imshow('GUI',image_gui)
        # cv2.imshow('Gray',image_gray)
        # cv2.imshow('ROI',image_roi)
    
        if cv2.waitKey(35) & 0xFF == ord('q') :
            break

        frame_number += 1

    
if __name__ == "__main__":
    main()
