#!/usr/bin/env python3

import pickle
import random
from colorama import Fore, Style
import matplotlib.pyplot as plt
import numpy as np
from line_model import LineModel


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    # Setup matplotlib figure
    plt.title('Ex2', fontweight="bold")
    plt.axis([-10, 10, -5, 5])

    # Load the points from file
    with open('points.pkl', 'rb') as f:
        points = pickle.load(f)

    # Draw points on figure
    plt.plot(points['xs'], points['ys'], 'rx')

    # Define a line model
    line = LineModel()
    best_line = LineModel()
    min_error = 10000000

    # --------------------------------------
    # Execution
    # --------------------------------------
    while True:

        # Create new line parameters
        line.randomizeParams()

        error = line.computeError(xs=points['xs'],
                                  ys_ground_truth=points['ys'])
        print('error = ' + str(error))
        # print('m = ' + str(line.m))

        if error < min_error:  # new best line found
            print(Fore.RED + 'Found a better model!' + Style.RESET_ALL)
            best_line.m = line.m  # update of the best model parameters
            best_line.b = line.b
            min_error = error

        # Draw new line
        line.draw(plt)

        # Draw the best line in red
        best_line.draw(plt, color='r')

        # Draw figure
        plt.draw()
        pressed_key = plt.waitforbuttonpress(0.1)
        if pressed_key == True:
            break

    # --------------------------------------
    # Termination
    # --------------------------------------

    with open('points.pkl', 'wb') as f:  # open a text file
        pickle.dump(points, f)  # serialize the list


if __name__ == "__main__":
    main()
