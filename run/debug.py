#!/usr/bin/env python

import subprocess, re, time, sys

done = False
count = 1

while not done:
    print('Starting run %d...' % count, end=" ", flush=True)
    start = time.time()
    p = subprocess.Popen('./run.sh',
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    output = p.stdout.read().decode()
    end = time.time()

    re_scientific_not = '-?[\d.]+(?:E-?\d+)?'
    pattern = 'Error = ' + re_scientific_not
    piece = re.findall(pattern, output)[0]
    error = re.findall(re_scientific_not, piece)[-1]
    error = float(error)

    if error !=  3.9736675973:
        done = True
        print(output)
    else:
        print('Run %d was correct. Took %f s' % (count, end-start))
        count += 1
