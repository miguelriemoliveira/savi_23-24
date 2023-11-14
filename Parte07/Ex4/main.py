#!/usr/bin/env python3

import pickle
import random
from colorama import Fore, Style
import matplotlib.pyplot as plt
import numpy as np
from line_model import LineModel

from scipy.optimize import least_squares


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

    # --------------------------------------
    # Execution
    # --------------------------------------

    def objectiveFunction(params):

        line.m = params[0]
        line.b = params[1]

        error = line.computeError(
            xs=points['xs'], ys_ground_truth=points['ys'])
        print('error = ' + str(error))

        # Draw new line
        line.draw(plt)

        # Draw figure
        plt.draw()
        pressed_key = plt.waitforbuttonpress(0.1)
        if pressed_key == True:
            exit(0)

        return error

    params0 = [line.m, line.b]
    result = least_squares(objectiveFunction, params0, verbose=2)

    print(result)

    # draw final solution
    line.m = result.x[0]
    line.b = result.x[1]

    line.draw(plt, color='g')
    plt.draw()
    pressed_key = plt.waitforbuttonpress(0.1)
    plt.show()

    # --------------------------------------
    # Termination
    # --------------------------------------


if __name__ == "__main__":
    main()
