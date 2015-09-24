#!/usr/bin/python

import random
import sys
import numpy as np

def sampling(path, rate):
    fp = open(path, 'r')
    fp_train = open(path + '.train', 'w')
    fp_test= open(path + '.test', 'w')
    while True:
        line = fp.readline()
        if len(line) == 0:
            break
        choose = random.random()
        rate = float(rate)
        if choose < rate:
            fp_test.write(line)
        else:
            fp_train.write(line)
    fp.close()
    fp_test.close()
    fp_train.close()


file_path = sys.argv[1]
rate = sys.argv[2]
sampling(file_path, rate)

