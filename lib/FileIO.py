#! /usr/bin/python

import csv


def csv_reader(file_name):
    cf = open(file_name)
    reader = csv.DictReader(cf)
    return reader


def read_lines(file_name, start_line):
    f = open(file_name)
    lines = f.readlines()
    f.close()
    return lines[start_line:]


def save_lines(file_name, lines):
    f = open(file_name, 'w')
    f.writelines(lines)
    f.close()
    return len(lines)
