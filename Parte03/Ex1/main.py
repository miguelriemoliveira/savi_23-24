#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA

import copy
import time

import cv2
import numpy as np


def main():

    #Lane 0
    # point_start = (687, 362)
    # point_end = (847, 515)

    #Lane 1             
    # point_start = (536, 285)
    # point_end = (639, 414)

    # Lane 3
    point_start = (799, 219)
    point_end = (934, 318)


    # Load the image
    # scene = cv2.imread('../images/scene.jpg') # relative path
    cap = cv2.VideoCapture('../docs/traffic.mp4')

    frame_number = 0
    average = 134
    num_cars = 0
    change_detected = False
    previous_change_detected = False
    tic = time.time()

    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, image_rgb = cap.read()
        if ret is False:
            break

        image_gui = copy.deepcopy(image_rgb)

        # Convert to gray
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # Get sub_image
        image_roi = image_gray[point_start[1]:point_end[1] , point_start[0]:point_end[0]]
        average_previous = average
        average = np.mean(image_roi)
        
        # Rising edge Method
#         t = 10.0
#         previous_change_detected = change_detected
#         if abs(average - average_previous) > t:
#             print('Change Detected!')
#             change_detected = True
# 
#         else:
#             change_detected = False
# 
#         # Rising edge detector
#         if previous_change_detected == False and change_detected == True: 
#             num_cars += 1 # Assume a change as a new car
#             pass
        
        # Zé
        # t = 10.0
        # previous_change_detected = change_detected
        # if abs(average - average_previous) > t:
        #     print('Change Detected!')
        #     change_detected = True
        #     cv2.waitKey(1000)
        # else:
        #     change_detected = False

        
        # Blackout
        t = 10.0
        blackout_threshold = 1.0
        previous_change_detected = change_detected
        time_since_tic = time.time() - tic

        if abs(average - average_previous) > t and time_since_tic > blackout_threshold:
            print('Change Detected!')
            change_detected = True
            tic = time.time()
            num_cars += 1 # Assume a change as a new car
            
        else:
            change_detected = False


        # --------------------------------------
        # Visualization
        # --------------------------------------

        cv2.rectangle(image_gui, (point_start[0], point_start[1]), (point_end[0], point_end[1]), (0,255,0), 4)

        image_gui = cv2.putText(image_gui, 'Frame ' + str(frame_number), (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255,255,0), 2, cv2.LINE_AA)
        image_gui = cv2.putText(image_gui, 'Avg ' + str(round(average,1)), (500, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0,255,255), 2, cv2.LINE_AA)
        image_gui = cv2.putText(image_gui, 'NCars ' + str(round(num_cars,1)), (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0,255,0), 2, cv2.LINE_AA)

        image_gui = cv2.putText(image_gui, 'TimeSinceTic ' + str(round(time_since_tic,1)), (500, 95), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0,255,0), 2, cv2.LINE_AA)


        cv2.imshow('GUI',image_gui)
        # cv2.imshow('Gray',image_gray)
        cv2.imshow('ROI',image_roi)
    
        if cv2.waitKey(35) & 0xFF == ord('q') :
            break

        frame_number += 1

    
if __name__ == "__main__":
    main()
