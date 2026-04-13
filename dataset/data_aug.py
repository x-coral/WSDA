from albumentations import *
import albumentations as albu
from random import randint
import numpy as np

def min_max(image, max, min):
    image_new = (image - np.min(image)) * (max - min) / ((np.max(image) - np.min(image)) + min + 1e-6)
    return image_new

def cropping(image, y, x, dim1, dim2):
    cropped_img = image[dim1:dim1+y, dim2:dim2+x]
    return cropped_img

def cut_and_paste(image1, label1_connections, image2, label2_connections, cut_size=(256,256)):
    size = image2.shape
    image2_y_loc = randint(0, size[0] - cut_size[0])
    image2_x_loc = randint(0, size[1] - cut_size[1])

    image2_cut = cropping(image2, cut_size[0], cut_size[1], image2_y_loc, image2_x_loc)
    label2_cut = cropping(label2_connections, cut_size[0], cut_size[1], image2_y_loc, image2_x_loc)

    size = image1.shape
    image1_y_loc = randint(0, size[0] - cut_size[0])
    image1_x_loc = randint(0, size[1] - cut_size[1])

    image1[image1_y_loc:image1_y_loc + cut_size[0], image1_x_loc:image1_x_loc + cut_size[1]] = image2_cut
    label1_connections[image1_y_loc:image1_y_loc + cut_size[0], image1_x_loc:image1_x_loc + cut_size[1]] = label2_cut

    return image1, label1_connections

def cut_and_paste_v1(image1, label1_list, image2, label2_list, cut_size=(256,256)):
    size = image2.shape
    image2_y_loc = randint(0, size[0] - cut_size[0])
    image2_x_loc = randint(0, size[1] - cut_size[1])

    image2_cut = cropping(image2, cut_size[0], cut_size[1], image2_y_loc, image2_x_loc)
    label2_cut_list = []
    for i in range(len(label2_list)):
        label2_cut_list.append(cropping(label2_list[i], cut_size[0], cut_size[1], image2_y_loc, image2_x_loc))
   
    size = image1.shape
    image1_y_loc = randint(0, size[0] - cut_size[0])
    image1_x_loc = randint(0, size[1] - cut_size[1])

    image1[image1_y_loc:image1_y_loc + cut_size[0], image1_x_loc:image1_x_loc + cut_size[1]] = image2_cut
    for i in range(len(label1_list)):
        label1_list[i][image1_y_loc:image1_y_loc + cut_size[0], image1_x_loc:image1_x_loc + cut_size[1]] = label2_cut_list[i]

    return image1, label1_list


def strong_aug(p=.5, cropsize=(512, 512)):
    return Compose([
        Flip(),
        Transpose(),
        RandomGamma(p=0.2),
        Rotate(),
        # RandomBrightnessContrast(),
        ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=0.5, hue=[0, 0.5]),

        # MotionBlur(p=0.2),
        GaussNoise(p=0.2, var_limit=(0, 0.3)),
        # GaussNoise(p=1.0, var_limit=(0, 0.5)),
        ElasticTransform(p=0.5),

        # Perspective(p=0.5),
        # Affine(p=0.5, scale=0.8, translate_percent={'x':(-0.3,0.3), 'y':(-0.3, 0.3)}, rotate=30, cval=0, shear=30),

        OneOrOther(
            RandomCrop(height=cropsize[0], width=cropsize[1], always_apply=True),
            RandomResizedCrop(height=cropsize[0], width=cropsize[1], always_apply=True),
            # RandomCrop(height=cropsize[0], width=cropsize[1], always_apply=True),
            p=0.9
        ),
        # CoarseDropout(max_holes=3, max_height=8, max_width=8, min_holes=1, fill_value=0.5, p=0.5)
    ], p=p)
    
    
def strong_aug_exitcrop(p=.5, cropsize=(512, 512)):
    return Compose([
        Flip(),
        Transpose(),
        RandomGamma(p=0.2),
        Rotate(),
        # RandomBrightnessContrast(),
        ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=0.5, hue=[0, 0.5]),
        # MotionBlur(p=0.2),
        GaussNoise(p=0.2, var_limit=(0, 0.3)),
        # GaussNoise(p=1.0, var_limit=(0, 0.5)),
        ElasticTransform(p=0.2),

        # CropNonEmptyMaskIfExists(height=cropsize[0], width=cropsize[1], always_apply=True),
        OneOrOther(
            CropNonEmptyMaskIfExists(height=cropsize[0], width=cropsize[1], always_apply=True, ignore_values=[0,255]),
            # RandomCrop(height=cropsize[0], width=cropsize[1], always_apply=True),
            RandomResizedCrop(height=cropsize[0], width=cropsize[1], always_apply=True),
            p=0.9,
        ),

        # CoarseDropout(max_holes=2, max_height=16, max_width=16, min_holes=1, fill_value=0, p=0.5)

    ], p=p)

def aug_img_lab(img, lab, cropsize, p=0.5):
    images = [img, lab]
    transformations = [strong_aug(p=p, cropsize=cropsize)]
    target = {'mask': 'mask'}
    transformed = albu.Compose(transformations, p=0.8, additional_targets=target)(image=images[0],
                                                                           mask=images[1],
                                                                           )
    return transformed['image'], transformed['mask']


def aug_target_img_lab(img, lab, cropsize, p=0.5):
    images = [img, lab]
    transformations = [strong_aug_exitcrop(p=p, cropsize=cropsize)]
    target = {'mask': 'mask'}
    transformed = albu.Compose(transformations, p=0.8, additional_targets=target)(image=images[0],
                                                                           mask=images[1]
                                                                           )
    return transformed['image'], transformed['mask']

def aug_target_img_lab3(img, lab1, lab2, lab3, cropsize, p=0.5):
    images = [img, lab1, lab2, lab3]
    transformations = [strong_aug_exitcrop(p=p, cropsize=cropsize)]
    target = {'mask': 'mask', 'mask1': 'mask', 'bg_mask': 'mask'}
    transformed = albu.Compose(transformations, p=0.8, additional_targets=target)(image=images[0],
                                                                           mask=images[1], mask1=images[2], bg_mask=images[3]
                                                                           )
    return transformed['image'], transformed['mask'], transformed['mask1'], transformed['bg_mask']