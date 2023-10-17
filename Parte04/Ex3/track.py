#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA

import copy
import csv
import time

import cv2
import numpy as np


class Detection():
    def __init__(self, left, right, top, bottom, id):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.detector_id = id

    def draw(self, image, color, draw_position='bottom', text=None):
        start_point = (self.left, self.top)
        end_point = (self.right, self.bottom)
        cv2.rectangle(image, start_point, end_point, color, 3)

        if text is None:
            text = 'Det ' + self.detector_id

        if draw_position == 'bottom':
            position = (self.left, self.bottom + 30)
        else:
            position = (self.left, self.top-10)

        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    def getLowerMiddlePoint(self):
        return (self.left + int((self.right - self.left)/2) , self.bottom)


class Track():

    # Class constructor
    def __init__(self, id, detection,  color=(255, 0, 0)):
        self.tracker_id = id
        self.color = color
        self.detections = [detection]

        print('Starting constructor for track id ' + str(self.tracker_id))

    def draw(self, image):

        #Draw only last detection
        self.detections[-1].draw(image, self.color, text=self.tracker_id, draw_position='top')

        for detection_a, detection_b in zip(self.detections[0:-1], self.detections[1:]):
            start_point = detection_a.getLowerMiddlePoint()
            end_point = detection_b.getLowerMiddlePoint()
            cv2.line(image, start_point, end_point, self.color, 3) 


    def update(self, left, right, top, bottom):

        self.detections.append(Detection(left,right, top, bottom))

#     def __repr__(self):
#         left, right, top, bottom  = self.detections[-1] # Get the last know position, i.e. last detection in the list of detections
# 
#         return 'track ' + str(self.id) + ' ndets=' + str(len(self.detections)) + ' l=' + str(left) + ' r=' + str(right) + ' t=' + str(top) + ' b=' + str(bottom)
#         