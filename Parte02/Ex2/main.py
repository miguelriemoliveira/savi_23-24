#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA

import copy

import cv2


def main():

    # Load the image
    scene = cv2.imread('../images/scene.jpg') # relative path
    template = cv2.imread('../images/wally.png') # relative path

    # --------------------------------------
    # Find Wally
    # --------------------------------------
    result = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)

    _, value_max, _, max_loc = cv2.minMaxLoc(result)
    print(value_max)
    print(max_loc)

    h,w,_ = template.shape
    cv2.rectangle(scene, (max_loc[0], max_loc[1]), (max_loc[0]+w, max_loc[1]+h), (0,255,0), 4)


    # --------------------------------------
    # Visualization
    # --------------------------------------
    cv2.imshow('Scene', scene)
    cv2.imshow('Template', template)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
