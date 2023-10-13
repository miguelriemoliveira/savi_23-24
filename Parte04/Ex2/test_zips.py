#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA

import copy
import csv
import time

import cv2
import numpy as np
from colorama import Fore, Back, Style


def main():

    A = [(0,0,0,0), (1,1,1,1), (2,2,2,2), (3,3,3,3), (4,4,4,4), (5,5,5,5)]

    primeiros = A[0:-1]
    segundos = A[1:]

    print('A = ' + str(A))
    print('primeiros = ' + str(primeiros))
    print('segundos = ' + str(segundos))

    i = 0
    for primeiro, segundo in zip(primeiros, segundos):

        
        print('Iteração ' + str(i) + ' primeiro=' + str(primeiro) + ' segundo=' + str(segundo))
        i += 1
    
if __name__ == "__main__":
    main()
