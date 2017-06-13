#! /usr/bin/python

import getopt
import sys


def deal_args(my_argv):
    v, d, n1, n0, b, e, t, z = False, False, 200, 200, 30, 1000, 8, 1000
    m = 'lenet'
    try:
        opts, args = getopt.getopt(my_argv, "vdhy:n:b:e:t:z:m:",
                                   ["p_sample_size=", "n_sample_size=", "batch_size=", "epoch_num=", "thread_num=",
                                    'test_size=', 'network_model='])
    except getopt.GetoptError:
        print 'DL_MS.py -v -d -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, ' \
              '-z <test_size>, -m <network_model>'
        print 'default settings: v=%s, d=%s, n1=%d, n0=%d, b=%d, e=%d, t=%d, z=%d, m=%s' % (
            v, d, n1, n0, b, e, t, z, m)
    for opt, arg in opts:
        if opt == '-h':
            print 'DL_MS.py -v -d -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, ' \
                  '-z <test_size>, -m <network_model>'
            sys.exit()
        elif opt == '-d':
            d = True
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
        elif opt in ("-z", "--test_size"):
            z = int(arg)
        elif opt in ("-m", "--network_model"):
            m = arg
    print 'settings: v=%s, d=%s, n1=%d, n0=%d, b=%d, e=%d, t=%d, z=%d, m=%s' % (
        v, d, n1, n0, b, e, t, z, m)
    return v, d, n1, n0, b, e, t, z, m


def deal_args_active(my_argv):
    v, d, n1, n0, b, e, t, z, a, t_up, t_low = False, False, 200, 200, 30, 1000, 8, 1000, 50, 0.55, 0.45
    m = 'lenet'
    try:
        opts, args = getopt.getopt(my_argv, "vdhy:n:b:e:t:z:m:a:u:l:",
                                   ["p_sample_size=", "n_sample_size=", "batch_size=", "epoch_num=", "thread_num=",
                                    'test_size=', 'network_model=', 'active_size=', 'threshold_up=', 'threshold_low='])
    except getopt.GetoptError:
        print 'DL_MS.py -v -d -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, ' \
              '-z <test_size>, -m <network_model>, -a <active_size>, -u <threshold_up>, -l <threshold_low>'
        print 'default settings: v=%s, d=%s, n1=%d, n0=%d, b=%d, e=%d, t=%d, z=%d, m=%s, a=%d, u=%f, l=%f' % (
            v, d, n1, n0, b, e, t, z, m, a, t_up, t_low)
    for opt, arg in opts:
        if opt == '-h':
            print 'DL_MS.py -v -d -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, ' \
                  '-z <test_size>, -m <network_model>, -a <active_size>, -u <threshold_up>, -l <threshold_low>'
            sys.exit()
        elif opt == '-d':
            d = True
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
        elif opt in ("-z", "--test_size"):
            z = int(arg)
        elif opt in ("-m", "--network_model"):
            m = arg
        elif opt in ("-a", "--active_size"):
            a = int(arg)
        elif opt in ("-u", "--threshold_up"):
            u = float(arg)
        elif opt in ("-l", "--threshold_low"):
            print arg
            l = float(arg)
    print 'settings: v=%s, d=%s, n1=%d, n0=%d, b=%d, e=%d, t=%d, z=%d, m=%s, a=%d, u=%f, l=%f' % (
        v, d, n1, n0, b, e, t, z, m, a, u, l)
    return v, d, n1, n0, b, e, t, z, m, a, u, l
