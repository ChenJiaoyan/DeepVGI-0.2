#! /usr/bin/python

import getopt
import sys


def deal_args(my_argv):
    v, n1, n0, b, e, t, c, z = False, 200, 200, 30, 1000, 8, 0, 1000
    m = 'lenet'
    try:
        opts, args = getopt.getopt(my_argv, "vhy:n:b:e:t:c:z:m:",
                                   ["p_sample_size=", "n_sample_size=", "batch_size=", "epoch_num=", "thread_num=",
                                    "cv_round=", 'test_size=', 'network_model='])
    except getopt.GetoptError:
        print 'DL_MS.py -v -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, ' \
              '-c <cv_round>, -z <test_size>, -m <network_model>'
        print 'default settings: v=%s, n1=%d, n0=%d, b=%d, e=%d, t=%d, c=%d, z=%d, m=%s' % (v, n1, n0, b, e, t, c, z, m)
    for opt, arg in opts:
        if opt == '-h':
            print 'DL_MS.py -v -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, ' \
                  '-c <cv_round>, -z <test_size>, -m <network_model>'
            sys.exit()
        elif opt == '-v':
            v = True
        elif opt in ("-y", "--p_sample_size"):
            n1 = int(arg)
        elif opt in ("-n", "--n_sample_size"):
            n0 = int(arg)
        elif opt in ("-b", "--batch_size"):
            b = int(arg)
        elif opt in ("-e", "--epoch_num"):
            e = int(arg)
        elif opt in ("-t", "--thread_num"):
            t = int(arg)
        elif opt in ("-c", "--cv_round"):
            c = int(arg)
        elif opt in ("-z", "--test_size"):
            z = int(arg)
        elif opt in ("-m", "--network_model"):
            m = arg
    print 'settings: v=%s, n1=%d, n0=%d, b=%d, e=%d, t=%d, c=%d, z=%d, m=%s' % (v, n1, n0, b, e, t, c, z, m)
    return v, n1, n0, b, e, t, c, z, m
