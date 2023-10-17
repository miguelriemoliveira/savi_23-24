#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA

import copy
import csv
import time
from random import randint

import cv2
import numpy as np
from track import Detection, Track
from colorama import Fore, Back, Style


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------
    cap = cv2.VideoCapture('../docs/OxfordTownCentre/TownCentreXVID.mp4')

    file ='../docs/OxfordTownCentre/TownCentre-groundtruth.top'
    gt_tracks = csv.reader(open(file))

    # Create person detector
    detector_filename = './fullbody.xml' 
    detector = cv2.CascadeClassifier(detector_filename)

    # Load a test image just for testing the detector
#     image_rgb = cv2.imread('./example_person.jpg')
#     cv2.namedWindow('example_person',cv2.WINDOW_NORMAL)
#     cv2.imshow('example_person',image_rgb)
# 
#     # Convert to grayscale to feed the detector
#     image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
#     cv2.namedWindow('example_person_gray',cv2.WINDOW_NORMAL)
#     cv2.imshow('example_person_gray',image_gray)

    video_frame_number = 0
    tracks = {}

    # --------------------------------------
    # Execution
    # --------------------------------------
    while(cap.isOpened()): # iterate video frames

        
        result, image_rgb = cap.read() # Capture frame-by-frame
        if result is False:
            break

        height, width, _ = image_rgb.shape
        image_gui = copy.deepcopy(image_rgb) # good practice to have a gui image for drawing

    
        # ------------------------------------------------------
        # Detect persons using haar cascade classifier
        # ------------------------------------------------------
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        haar_detections = detector.detectMultiScale(image_gray, scaleFactor=1.05, minNeighbors=5,
                                            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # ------------------------------------------------------
        # Create list of detections
        # ------------------------------------------------------
        detections = []
        detection_idx = 0
        for x,y,w,h in haar_detections:
            detection_id = str(video_frame_number) + '_' +  str(detection_idx)
            detection = Detection(x, x+w, y, y+h, detection_id)
            detections.append(detection)
            detection_idx += 1


               
        # --------------------------------------
        # Visualization
        # --------------------------------------

        # Draw list of detections
        for detection in detections:
            detection.draw(image_gui, (255,0,0))



        cv2.namedWindow('GUI',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('GUI', int(width/2), int(height/2))
        cv2.imshow('GUI',image_gui)
            
        if cv2.waitKey(25) & 0xFF == ord('q') :
            break

        video_frame_number += 1

    
if __name__ == "__main__":
    main()
