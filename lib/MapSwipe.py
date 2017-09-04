#! /usr/bin/python

import math
import FileIO
import os


def cal_lat_lon(task_x, task_y):
    task_z = 18
    PixelX = task_x * 256
    PixelY = task_y * 256
    MapSize = 256 * math.pow(2, task_z)
    x = (PixelX / MapSize) - 0.5
    y = 0.5 - (PixelY / MapSize)
    lon_left = 360 * x
    lat_top = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi

    PixelX = (task_x + 1) * 256
    PixelY = (task_y + 1) * 256
    MapSize = 256 * math.pow(2, task_z)
    x = (PixelX / MapSize) - 0.5
    y = 0.5 - (PixelY / MapSize)
    lon_right = 360 * x
    lat_bottom = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi

    return lat_top, lon_left, lat_bottom, lon_right


def cal_pixel(lat, lon):
    task_z = 18
    sin_lat = math.sin(lat * math.pi / 180)
    x = ((lon + 180) / 360) * 256 * math.pow(2, task_z)
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * 256 * math.pow(2, task_z)
    task_x = int(math.floor(x / 256))
    task_y = int(math.floor(y / 256))
    pixel_x = int(x % 256 + 0.5)
    pixel_y = int(y % 256 + 0.5)
    return task_x, task_y, pixel_x, pixel_y


class MSClient:
    def __init__(self):
        self.sample_dir = '../samples0/'

    def MS_train_record(self):
        return os.listdir(os.path.join(self.sample_dir, 'train/MS_record'))

    def MS_train_negative(self):
        return os.listdir(os.path.join(self.sample_dir, 'train/MS_negative'))

    def MS_valid_record(self):
        return os.listdir(os.path.join(self.sample_dir, 'valid/MS_record'))

    def MS_valid_negative(self):
        return os.listdir(os.path.join(self.sample_dir, 'valid/MS_negative'))

    def MS_positive(self):
        imgs = []
        ms_file = '../data/project_922.csv'
        lines = FileIO.read_lines(ms_file, 1)
        for line in lines:
            tmp = line.strip().split(',')
            x, y = int(tmp[4]), int(tmp[5])
            img = '%d-%d.jpeg' % (x, y)
            yes_count, maybe_count, bad_img_count = int(tmp[8]), int(tmp[9]), int(tmp[10])
            if bad_img_count == 0 and (yes_count >= 2 or (maybe_count + yes_count) >= 3):
                imgs.append(img)
        return imgs

    def MS_train_positive(self):
        record = os.listdir(os.path.join(self.sample_dir, 'train/MS_record'))
        return list(set(record).intersection(set(self.MS_positive())))

    def MS_valid_positive(self):
        record = os.listdir(os.path.join(self.sample_dir, 'valid/MS_record'))
        return list(set(record).intersection(set(self.MS_positive())))

class Urban_client():     # only valid images, ms positive + urban extent
    def __init__(self):
        self.sample_dir = '../samples0/'

    def valid_negative(self):
        return os.listdir(os.path.join(self.sample_dir, 'valid/MS_negative'))

    def urban_positive(self):
        urban_file = '../data/malawi_urban_extent.csv'
        lines = FileIO.csv_reader(urban_file)
        p_imgs_raw = []
        for line in lines:
            task_x = line['task_x']
            task_y = line['task_y']
            c = line['classification']
            if c == 'urban_extent':
                img = '%s-%s.jpeg' % (task_x, task_y)
                p_imgs_raw.append(img)
        p_imgs = list(set(p_imgs_raw))
        return p_imgs

    def valid_positive(self):
        ms = MSClient()
        ms_p = ms.MS_valid_positive()
        return list(set(ms_p).intersection(set(self.urban_positive())))