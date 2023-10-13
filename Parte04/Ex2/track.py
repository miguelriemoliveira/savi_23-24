#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA

import copy
import csv
import time

import cv2
import numpy as np


class Track():

    # Class constructor
    def __init__(self, id, left, right, top, bottom):
        self.id = id
        self.detections = [(left, right, top, bottom)]

        print('Starting constructor for track id ' + str(self.id))

    def draw(self, image):

        for detection in self.detections: 

            left, right, top, bottom  = detection
            start_point = (left, top)
            end_point = (right, bottom)
            cv2.rectangle(image, start_point, end_point, (0,255,0), 3)

            image_gui = cv2.putText(image, 't' + str(self.id), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    def update(self, left, right, top, bottom):

        self.detections.append((left,right, top, bottom))

    def __repr__(self):
        left, right, top, bottom  = self.detections[-1] # Get the last know position, i.e. last detection in the list of detections

        return 'track ' + str(self.id) + ' ndets=' + str(len(self.detections)) + ' l=' + str(left) + ' r=' + str(right) + ' t=' + str(top) + ' b=' + str(bottom)
        