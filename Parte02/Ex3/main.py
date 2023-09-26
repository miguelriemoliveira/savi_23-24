#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA

import copy

import cv2

point_start = None
point_end = None

def mouseCallback(event,x,y,flags,param):
    global point_start, point_end

    if event == cv2.EVENT_LBUTTONDOWN:
        point_start = (x,y)
        print('recorded point start')
    elif event == cv2.EVENT_LBUTTONUP:
        point_end = (x,y)
        print('recorded point end')

def main():

    global point_start, point_end

    # Load the image
    scene = cv2.imread('../images/scene.jpg') # relative path


    # --------------------------------------
    # Define new template
    # --------------------------------------

    cv2.imshow('Scene', scene)
    cv2.setMouseCallback('Scene',mouseCallback)

    while True: # waiting for template definition
        if point_start is not None and point_end is not None:
            break
        cv2.waitKey(20)

    print('point_start = ' + str(point_start))
    print('point_end = ' + str(point_end))


    template = scene[point_start[1]:point_end[1], point_start[0]:point_end[0]] # relative path

    cv2.imshow('Template', template)
    cv2.waitKey(2)

    # --------------------------------------
    # Find Wally
    # --------------------------------------
    result = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)

    _, value_max, _, max_loc = cv2.minMaxLoc(result)
    print(value_max)
    print(max_loc)

    h,w,_ = template.shape
    cv2.rectangle(scene, (max_loc[0], max_loc[1]), (max_loc[0]+w, max_loc[1]+h), (0,255,0), 4)

    cv2.imshow('Search', scene)
    cv2.waitKey(0)


    # --------------------------------------
    # Visualization
    # --------------------------------------
    
if __name__ == "__main__":
    main()
