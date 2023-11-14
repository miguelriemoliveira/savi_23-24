#!/usr/bin/env python3

import pickle
import random
import matplotlib.pyplot as plt
import numpy as np


class LineModel():

    def __init__(self):

        self.m = 0.1
        self.b = 3
        self.handle = None

    def computeError(self, xs, ys_ground_truth):

        ys = self.getYs(xs)

        total_error = 0
        for y, y_ground_truth in zip(ys, ys_ground_truth):
            error = abs(y_ground_truth-y)
            total_error += error

        return total_error

    def randomizeParams(self):
        self.m = random.uniform(-2, 2)
        self.b = random.uniform(-15, 15)

    def getYs(self, xs):
        ys = []
        for x in xs:
            ys.append(self.m * x + self.b)

        return ys

    def draw(self, fig, color='b'):

        xs = list(np.linspace(-10, 10, num=2))
        ys = self.getYs(xs)

        if self.handle is None:  # draw first time
            self.handle = fig.plot(xs, ys, '-' + color)

        else:  # edit plot all other times
            fig.setp(self.handle, xdata=xs, ydata=ys)
