import cv2
import numpy as np
from skimage import measure
from skimage import morphology as morph

def generate_center_point(binary_map):

    binary_map = binary_map.astype('uint8')

    labels = measure.label(binary_map, connectivity=1)

    properties = measure.regionprops(labels)

    gt = np.zeros_like(binary_map).astype('float')
    for prop in properties:
        # print(int(prop.centroid[0]), int(prop.centroid[1]))
        gt[int(prop.centroid[0]), int(prop.centroid[1])] = 1

    return gt


def make_gaussian_map(point_map, sigma):

    count_map = cv2.GaussianBlur(point_map.astype('float32'), sigma, 0, borderType=0)
    am = np.amax(count_map)
    if am != 0:
        count_map /=am/255.0
    else:
        count_map = count_map

    return count_map

def generate_center_gaussianmap(binary_map, sigma):

    binary_map = binary_map.astype('uint8')

    labels = measure.label(binary_map, connectivity=1)

    properties = measure.regionprops(labels)

    gt = np.zeros_like(binary_map).astype('float')
    for prop in properties:
        # print(int(prop.centroid[0]), int(prop.centroid[1]))
        gt[int(prop.centroid[0]), int(prop.centroid[1])] = 1

    count_map = cv2.GaussianBlur(gt, sigma, 0, borderType=0)
    am = np.amax(count_map)
    if am != 0:
        count_map /= am/255.0
    else:
        count_map = count_map

    return gt, count_map