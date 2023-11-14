#!/usr/bin/env python3

import pickle
import matplotlib.pyplot as plt
import numpy as np


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    plt.title('Select points', fontweight="bold")
    plt.axis([-10, 10, -5, 5])

    # --------------------------------------
    # Execution
    # --------------------------------------
    print("Start selecting points")
    points = {'xs': [], 'ys': []}

    while True:
        click = plt.ginput(1)

        if not click:
            break

        points['xs'].append(click[0][0])
        points['ys'].append(click[0][1])
        print('points = ' + str(points))

        plt.plot(points['xs'], points['ys'], 'rx')

        plt.draw()
        plt.waitforbuttonpress(0.1)

    # --------------------------------------
    # Termination
    # --------------------------------------

    with open('points.pkl', 'wb') as f:  # open a text file
        pickle.dump(points, f)  # serialize the list


if __name__ == "__main__":
    main()
