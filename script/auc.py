#!/usr/bin/python

from __future__ import division
import sys
import math
import numpy as np

def auc(x, y):
    num_zeros = x.count(0)
    x = np.array(x)
    y = np.array(y)
    order = np.lexsort((y,x))
    x, y = x[order], y[order]
    count = 0
    for i in y[:num_zeros]:
        for j in y[num_zeros:]:
            if i < j:
                count += 1
    #print(count, num_zeros, len(x)-num_zeros)
    print "%.6f" % (count/(num_zeros*(len(x)-num_zeros)))

for file in sys.argv[1:]:
    fp = open(file, 'r')
    x = fp.readline()
    x = x[:-2]
    y = fp.readline()
    y = y[:-1]
    x = x.split(' ')
    x = [ float(i) for i in x ]
    y = y.split(' ')
    y = [ float(i) for i in y ]
    #print(x, y)
    auc(x,y)
