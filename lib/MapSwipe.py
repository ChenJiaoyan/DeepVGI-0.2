#! /usr/bin/python

import math
import FileIO
import os
import random


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
    def __init__(self, project_id=922, name='Malawi'):
        self.project_id = project_id
        self.name = name

    def read_p_images(self):
        ms_file = '../data/project_' + str(self.project_id) + '.csv'
        lines = FileIO.read_lines(ms_file, 1)
        p_imgs = []
        for line in lines:
            tmp = line.strip().split(',')
            x, y = int(tmp[4]), int(tmp[5])
            yes_count, maybe_count, bad_img_count = int(tmp[8]), int(tmp[9]), int(tmp[10])
            if bad_img_count == 0 and (yes_count >= 2 or (maybe_count + yes_count) >= 3):
                p_imgs.append([x, y])
        return p_imgs

    def read_n_images(self):
        img_dir = '../data/image_project_' + str(self.project_id) + '_negative/'
        imgs = os.listdir(img_dir)
        n_imgs = []
        for img in imgs:
            i1, i2 = img.index('-'), img.index('.')
            task_x, task_y = int(img[0:i1]), int(img[(i1 + 1):i2])
            n_imgs.append([int(task_x), int(task_y)])
        return n_imgs

    def imgs_cross_validation(self, cv_i, cv_n):
        img_dir1 = '../data/image_project_' + str(self.project_id) + '/'
        imgs1 = os.listdir(img_dir1)
        img_dir2 = '../data/image_project_' + str(self.project_id) + '_negative/'
        imgs2 = os.listdir(img_dir2)
        imgs = imgs1 + imgs2
        imgs = random.shuffle(imgs)
        l = len(imgs)
        batch = l / cv_n
        test_imgs = imgs[cv_i * batch: (cv_i + 1) * batch]
        train_imgs = imgs[0:cv_i * batch] + imgs[(cv_i + 1) * batch: l]
        return train_imgs, test_imgs
