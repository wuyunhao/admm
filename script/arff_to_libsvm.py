#!/usr/bin/python

import random
import sys
import re

def Formatting(path):
    fp2 = open(path + '.libsvm', 'w')
    fp = open(path, 'r')

    while(True):
        line = fp.readline()
        if len(line) == 0:
            break
        line = line.split(',')
        if len(line) < 3:
            continue
        label = re.split(r'[\n\r]',line[-1])
        fp2.write(label[0])
        count = 0
        for item in line[:-1]:
            if int(item) > 0:
                ok = ':'.join([str(count), str(1)])
                fp2.write(' ')
                fp2.write(ok)
            count += 1
        fp2.write('\n')
    fp.close()
    fp2.close()

for file in sys.argv[1:]:
    Formatting(file)
print sys.argv, "is processed successfully"

    
