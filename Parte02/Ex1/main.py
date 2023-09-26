#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA

import copy

import cv2


def main():

    # Load the image
    image_original = cv2.imread('../images/lake.jpg') # relative path

    # Nightfall

    print(type(image_original))
    print(image_original.shape)
    h, w, nc = image_original.shape

    image = copy.deepcopy(image_original)
    reduction = 50
    for row in range(0,h): # cycle all rows
        for col in range(0,w): # cycle all cols
            image[row,col,0] = max(image[row,col,0] - reduction, 0) # blue channel
            image[row,col,1] = max(image[row,col,1] - reduction, 0) # green channel
            image[row,col,2] = max(image[row,col,2] - reduction, 0) # red channel 

    # --------------------------------------
    # Visualization
    # --------------------------------------
    cv2.imshow('Original', image_original)
    cv2.imshow('Nightfall', image)
    cv2.waitKey(0)




if __name__ == "__main__":
    main()
