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
    # scene = cv2.imread('../images/scene.jpg') # relative path
    cap = cv2.VideoCapture('../docs/traffic.mp4')

    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            # Display the resulting frame
            cv2.imshow('Scene',frame)
        
            # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q') :
            if True:
                break

    # --------------------------------------
    # Define new template
    # --------------------------------------

    cv2.imshow('Scene', frame)
    cv2.setMouseCallback('Scene',mouseCallback)

    while True: # waiting for template definition
        if point_start is not None and point_end is not None:
            break
        cv2.waitKey(20)

    print('point_start = ' + str(point_start))
    print('point_end = ' + str(point_end))

    cv2.rectangle(frame, (point_start[0], point_start[1]), (point_end[0], point_end[1]), (0,255,0), 4)

    cv2.imshow('Scene', frame)
    cv2.waitKey(0)


    # --------------------------------------
    # Visualization
    # --------------------------------------
    
if __name__ == "__main__":
    main()
