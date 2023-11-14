#!/usr/bin/env python3

import pickle
import random
from colorama import Fore, Style
import matplotlib.pyplot as plt
import numpy as np
from line_model import LineModel, ParabModel

from scipy.optimize import least_squares


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    # Setup matplotlib figure
    plt.title('E42', fontweight="bold")
    plt.axis([-10, 10, -5, 5])

    # Load the points from file
    with open('points.pkl', 'rb') as f:
        points = pickle.load(f)

    xs_line = []
    ys_line = []
    xs_parab = []
    ys_parab = []

    for x, y in zip(points['xs'], points['ys']):
        if x <= 0:
            xs_line.append(x)
            ys_line.append(y)
        else:
            xs_parab.append(x)
            ys_parab.append(y)

    # Draw points on figure
    plt.plot(xs_line, ys_line, 'rx')
    plt.plot(xs_parab, ys_parab, 'gx')

    # Define a line model
    line = LineModel()
    parab = ParabModel()

    # --------------------------------------
    # Execution
    # --------------------------------------

    def objectiveFunction(params):

        line.m = params[0]
        line.b = params[1]
        parab.a = params[2]
        parab.h = params[3]
        parab.k = params[4]

        error_line = line.computeError(xs=xs_line, ys_ground_truth=ys_line)
        error_parab = parab.computeError(xs=xs_parab, ys_ground_truth=ys_parab)

        # Draw models
        line.draw(plt, color='r')
        parab.draw(plt, color='g')

        # Draw figure
        plt.draw()
        pressed_key = plt.waitforbuttonpress(0.1)
        if pressed_key == True:
            exit(0)

        return error_line + error_parab

    params0 = [line.m, line.b, parab.a, parab.h, parab.k]
    result = least_squares(objectiveFunction, params0, verbose=2)

    print(result)

    # draw final solution

    line.draw(plt, color='g')
    plt.draw()
    pressed_key = plt.waitforbuttonpress(0.1)
    plt.show()

    # --------------------------------------
    # Termination
    # --------------------------------------


if __name__ == "__main__":
    main()
