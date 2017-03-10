#! /usr/bin/python

import sys

sys.path.append("../lib")
import FileIO

import numpy as np

L = 50
n = 50
buildings = FileIO.read_lines("../data/building_samples.csv", 1)
X = np.zeros((n, 2))

for building in buildings:
    if 'yes' in building:
        break;
    else:
        break;
