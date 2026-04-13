from torch.utils import data
import numpy as np
import os.path as osp
import random
from PIL import Image
from skimage import morphology
import cv2
import torch

from .data_aug import *
from .detmap import *

class sourceDataSet_train(data.Dataset):
    def __init__(self, root_img, root_label, list_path, max_iters=None, crop_size=(512, 512), sigma=61, batch_size=1):
        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size
        self.sigma = sigma

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) * batch_size / len(self.img_ids)))
        self.files = []

        self.ignore_index = {}

        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles1 = self.files[index]
        rand_index = random.randint(0, len(self.files) - 1)
        datafiles2 = self.files[rand_index]

        image1 = Image.open(datafiles1["img"])
        image1 = np.asarray(image1, np.float32)
        image2 = Image.open(datafiles2["img"])
        image2 = np.asarray(image2, np.float32)

        label1 = Image.open(datafiles1["label"])
        label1 = np.asarray(label1, np.float32)
        label2 = Image.open(datafiles2["label"])
        label2 = np.asarray(label2, np.float32)

        name1 = datafiles1["name"]

        if np.random.random() > 0.1:
            image1, label1 = cut_and_paste(image1, label1, image2, label2, cut_size=(256, 256))
            
        image1 = min_max(image1, max=1, min=0)
        image_as_np, label_as_np = aug_img_lab(image1, label1, self.crop_size)
        image_as_np = min_max(image_as_np, max=1, min=0)

        label_as_np = morphology.remove_small_objects(label_as_np.astype('bool'), min_size=4, connectivity=1)
        gt_num, _ = cv2.connectedComponents(label_as_np.astype('uint8'))
        gt_num_tensor = torch.from_numpy(np.asarray((gt_num - 1))).long()
        label_as_np[label_as_np > 0] = 1

        size = image_as_np.shape

        points_map_np, detection_map_np = generate_center_gaussianmap(label_as_np.astype('uint8'), sigma=(self.sigma, self.sigma))
      
        detectionmap_np = np.expand_dims(detection_map_np, axis=0)
        detectionmap_as_tensor = torch.from_numpy(detectionmap_np.astype("float32")).long()

        image_as_np = np.expand_dims(image_as_np, axis=0)
        image_as_tensor = torch.from_numpy(image_as_np.astype("float32")).float()

        label_as_tensor = torch.from_numpy(label_as_np.astype("float32")).long()

        return image_as_tensor, label_as_tensor, points_map_np, detectionmap_as_tensor, gt_num_tensor, name1