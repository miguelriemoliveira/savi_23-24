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


    # Search for persons with an M in the name and delete these elements of the list
    # for idx_name, name in enumerate(names):
    #     if 'm' in name:
    #         del names[idx_name]


    # Search for persons with an M in the name and add to a list of idxs_to_remove
    idxs_names_to_remove = []
    for idx_name, name in enumerate(names):
        if 'm' in name:
            idxs_names_to_remove.append(idx_name)

    print('idxs_names_to_remove = ' + str(idxs_names_to_remove))

    idxs_names_to_remove.reverse()
    print('idxs_names_to_remove = ' + str(idxs_names_to_remove))

    for idx in idxs_names_to_remove:
        del names[idx]

    print('names = ' + str(names))
    
if __name__ == "__main__":
    main()
