#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import cv2
import os
lab_dir = '/media/yjj/A/XS/data/MitoEM-R-new/parttrain/resizelab'
save_dir = '/media/yjj/A/XS/data/MitoEM-R-new/parttrain/resizelab1'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for file_name in sorted(os.listdir(lab_dir)):
    file_path = os.path.join(lab_dir, file_name)
    lab = cv2.imread(file_path, -1)
    number = int(file_name[2:6])
    # if number in [401,414,429,439,441,448,474,478,485,494]:
    if number % 20 == 0:
    # if number < 400:
        save_path = os.path.join(save_dir, file_name)
        print(file_name)
        cv2.imwrite(save_path, lab)
