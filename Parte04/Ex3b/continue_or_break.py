#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA

import copy
import csv
import math
import time
from random import randint

import cv2
import numpy as np
from track import Detection, Track
from colorama import Fore, Back, Style


def main():

    names = ['carolina', 'miguel', 'hugo', 'josé', 'emanuel']


    for idx_name, name in enumerate(names):

        if name == 'miguel':
            continue

        names[idx_name] = names[idx_name] + '_' + str(77)

        # if name == 'hugo':
        #     break

    print(names)
    
if __name__ == "__main__":
    main()
