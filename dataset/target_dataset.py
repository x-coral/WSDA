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


class targetDataSet_train_step1(data.Dataset):
    def __init__(self, root_img, root_gtpointlab, root_partialab, root_detbg, list_path, max_iters=None, iter_start=0, crop_size=[512, 512],
                  sigma=61, batch_size=1):

        self.root_img = root_img
        self.root_gtpointlab = root_gtpointlab
        self.root_partialab = root_partialab
        self.root_detbg = root_detbg
        self.list_path = list_path
        self.sigma = sigma

        self.crop_size = crop_size
        self.max_iters = max_iters

        self.iter = iter_start

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) * batch_size / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            gtpointlab_file = osp.join(self.root_gtpointlab, name)
            partialab_file = osp.join(self.root_partialab, name)
            detbg_file = osp.join(self.root_detbg, name)
            self.files.append({
                "img": img_file,
                "gtpointlab": gtpointlab_file,
                "partialab": partialab_file,
                "detbg": detbg_file,
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

        gtpointlab1 = Image.open(datafiles1["gtpointlab"])  #0,1
        gtpointlab1 = np.asarray(gtpointlab1, np.float32)
        gtpointlab2 = Image.open(datafiles2["gtpointlab"])
        gtpointlab2 = np.asarray(gtpointlab2, np.float32)

        flabel1 = Image.open(datafiles1["partialab"])  #0,1,255
        flabel1 = np.asarray(flabel1, np.float32)
        flabel2 = Image.open(datafiles2["partialab"])
        flabel2 = np.asarray(flabel2, np.float32)
        
        detbg1 = Image.open(datafiles1["detbg"])  #0,1
        detbg1 = np.asarray(detbg1, np.float32)
        detbg2 = Image.open(datafiles2["detbg"])
        detbg2 = np.asarray(detbg2, np.float32)
        
        lab_list1 = [gtpointlab1, flabel1, detbg1]
        if np.random.random() > 0.1:
            lab_list2 = [gtpointlab2, flabel2, detbg2]
            image1, lab_list1 = cut_and_paste_v1(image1, lab_list1, image2, lab_list2, cut_size=(256, 256))


        image1 = min_max(image1, max=1, min=0)
        image_as_np, flabe1_as_np, gtpointlab_as_np, detbg_as_np = aug_target_img_lab3(image1, lab_list1[1], lab_list1[0], lab_list1[2], self.crop_size)
        image_as_np = min_max(image_as_np, max=1, min=0)
       
        size = image_as_np.shape

        image_as_np = np.expand_dims(image_as_np, axis=0)
        image_as_tensor = torch.from_numpy(image_as_np.astype("float32")).float()
        
        gtpoint = generate_center_point(gtpointlab_as_np.astype('uint8'))
        gtpoints_as_tensor = torch.from_numpy(gtpoint.astype("float32")).float()

        tlabel_as_tensor = torch.from_numpy(gtpointlab_as_np.astype("float32")).long()

        flabe1_as_np1 = flabe1_as_np.copy()
        flabe1_as_np1[flabe1_as_np1 != 1] = 0
        fpoints_map_np, fdetection_map_np = generate_center_gaussianmap(flabe1_as_np1.astype('uint8'), sigma=(self.sigma, self.sigma))

        fpoints_as_tensor = torch.from_numpy(fpoints_map_np.astype("float32")).float()
        fdetection_map_np = np.expand_dims(fdetection_map_np, axis=0)
        fdetection_map_as_tensor = torch.from_numpy(fdetection_map_np.astype("float32")).float()
    
        return image_as_tensor, tlabel_as_tensor, gtpoints_as_tensor, \
                fpoints_as_tensor, fdetection_map_as_tensor, detbg_as_np, np.array(size)


class targetDataSet_train_step2(data.Dataset):
    def __init__(self, root_img, root_gtpointlab, root_partialab, root_detbg, list_path, max_iters=None, iter_start=0, crop_size=[512, 512],
                  sigma=61, batch_size=1):

        self.root_img = root_img
        self.root_gtpointlab = root_gtpointlab
        self.root_partialab = root_partialab
        self.root_detbg = root_detbg
        self.list_path = list_path
        self.sigma = sigma

        self.crop_size = crop_size
        self.max_iters = max_iters

        self.iter = iter_start

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) * batch_size / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            gtpointlab_file = osp.join(self.root_gtpointlab, name)
            partialab_file = osp.join(self.root_partialab, name)
            detbg_file = osp.join(self.root_detbg, name)
            self.files.append({
                "img": img_file,
                "gtpointlab": gtpointlab_file,
                "partialab": partialab_file,
                "detbg": detbg_file,
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

        gtpointlab1 = Image.open(datafiles1["gtpointlab"])  #0,1
        gtpointlab1 = np.asarray(gtpointlab1, np.float32)
        gtpointlab2 = Image.open(datafiles2["gtpointlab"])
        gtpointlab2 = np.asarray(gtpointlab2, np.float32)

        flabel1 = Image.open(datafiles1["partialab"])  #0,1,255
        flabel1 = np.asarray(flabel1, np.float32)
        flabel2 = Image.open(datafiles2["partialab"])
        flabel2 = np.asarray(flabel2, np.float32)
        
        detbg1 = Image.open(datafiles1["detbg"])  #0,1
        detbg1 = np.asarray(detbg1, np.float32)
        detbg2 = Image.open(datafiles2["detbg"])
        detbg2 = np.asarray(detbg2, np.float32)
        
        lab_list1 = [gtpointlab1, flabel1, detbg1]
        if np.random.random() > 0.5:
            lab_list2 = [gtpointlab2, flabel2, detbg2]
            image1, lab_list1 = cut_and_paste_v1(image1, lab_list1, image2, lab_list2, cut_size=(256, 256))


        image1 = min_max(image1, max=1, min=0)
        image_as_np, flabe1_as_np, gtpointlab_as_np, detbg_as_np = aug_target_img_lab3(image1, lab_list1[1], lab_list1[0], lab_list1[2], self.crop_size)
        image_as_np = min_max(image_as_np, max=1, min=0)
       
        size = image_as_np.shape

        image_as_np = np.expand_dims(image_as_np, axis=0)
        image_as_tensor = torch.from_numpy(image_as_np.astype("float32")).float()
        
        gtpoint = generate_center_point(gtpointlab_as_np.astype('uint8'))
        gtpoints_as_tensor = torch.from_numpy(gtpoint.astype("float32")).float()

        flabel_as_tensor = torch.from_numpy(flabe1_as_np.astype("float32")).long()

        flabe1_as_np1 = flabe1_as_np.copy()
        flabe1_as_np1[flabe1_as_np1 != 1] = 0
        fpoints_map_np, fdetection_map_np = generate_center_gaussianmap(flabe1_as_np1.astype('uint8'), sigma=(self.sigma, self.sigma))

        fpoints_as_tensor = torch.from_numpy(fpoints_map_np.astype("float32")).float()
        fdetection_map_np = np.expand_dims(fdetection_map_np, axis=0)
        fdetection_map_as_tensor = torch.from_numpy(fdetection_map_np.astype("float32")).float()
    
        return image_as_tensor, gtpoints_as_tensor, flabel_as_tensor, \
                fpoints_as_tensor, fdetection_map_as_tensor, detbg_as_np, np.array(size)


class targetDataSet_train_step1_uda(data.Dataset):
    def __init__(self, root_img, root_gtpointlab, root_partialab, root_det, list_path, max_iters=None, iter_start=0, crop_size=[512, 512],
                  sigma=61, batch_size=1):

        self.root_img = root_img
        self.root_gtpointlab = root_gtpointlab
        self.root_partialab = root_partialab
        self.root_det = root_det
        self.list_path = list_path
        self.sigma = sigma

        self.crop_size = crop_size
        self.max_iters = max_iters

        self.iter = iter_start

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) * batch_size / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            gtpointlab_file = osp.join(self.root_gtpointlab, name)
            partialab_file = osp.join(self.root_partialab, name)
            det_file = osp.join(self.root_det, name)
            self.files.append({
                "img": img_file,
                "gtpointlab": gtpointlab_file,
                "partialab": partialab_file,
                "det": det_file,
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

        gtpointlab1 = Image.open(datafiles1["gtpointlab"])  #0,1
        gtpointlab1 = np.asarray(gtpointlab1, np.float32)
        gtpointlab2 = Image.open(datafiles2["gtpointlab"])
        gtpointlab2 = np.asarray(gtpointlab2, np.float32)

        flabel1 = Image.open(datafiles1["partialab"])  #0,1,255
        flabel1 = np.asarray(flabel1, np.float32)
        flabel2 = Image.open(datafiles2["partialab"])
        flabel2 = np.asarray(flabel2, np.float32)
        
        det1 = Image.open(datafiles1["det"])  #0-255
        det1 = np.asarray(det1, np.float32)
        det2 = Image.open(datafiles2["det"])
        det2 = np.asarray(det2, np.float32)
        
        lab_list1 = [gtpointlab1, flabel1, det1]
        if np.random.random() > 0.1:
            lab_list2 = [gtpointlab2, flabel2, det2]
            image1, lab_list1 = cut_and_paste_v1(image1, lab_list1, image2, lab_list2, cut_size=(256, 256))

        image1 = min_max(image1, max=1, min=0)
        image_as_np, flabe1_as_np, gtpointlab_as_np, det_as_np = aug_target_img_lab3(image1, lab_list1[1], lab_list1[0], lab_list1[2], self.crop_size)
        image_as_np = min_max(image_as_np, max=1, min=0)
       
        size = image_as_np.shape

        image_as_np = np.expand_dims(image_as_np, axis=0)
        image_as_tensor = torch.from_numpy(image_as_np.astype("float32")).float()
        
        gtpoint = generate_center_point(gtpointlab_as_np.astype('uint8'))
        gtpoints_as_tensor = torch.from_numpy(gtpoint.astype("float32")).float()

        tlabel_as_tensor = torch.from_numpy(gtpointlab_as_np.astype("float32")).long()

        flabe1_as_np1 = flabe1_as_np.copy()
        flabe1_as_np1[flabe1_as_np1 != 1] = 0
        fpoints_map_np, fdetection_map_np = generate_center_gaussianmap(flabe1_as_np1.astype('uint8'), sigma=(self.sigma, self.sigma))

        fpoints_as_tensor = torch.from_numpy(fpoints_map_np.astype("float32")).float()
        fdetection_map_np = np.expand_dims(fdetection_map_np, axis=0)
        fdetection_map_as_tensor = torch.from_numpy(fdetection_map_np.astype("float32")).float()
        
        detbg_as_np = np.zeros_like(det_as_np)
        detbg_as_np[det_as_np == 0] = 1
    
        return image_as_tensor, tlabel_as_tensor, gtpoints_as_tensor, \
                fpoints_as_tensor, fdetection_map_as_tensor, detbg_as_np, np.array(size)


class targetDataSet_train_step2_uda(data.Dataset):
    def __init__(self, root_img, root_gtpointlab, root_partialab, root_det, list_path, max_iters=None, iter_start=0, crop_size=[512, 512],
                  sigma=61, batch_size=1):

        self.root_img = root_img
        self.root_gtpointlab = root_gtpointlab
        self.root_partialab = root_partialab
        self.root_det = root_det
        self.list_path = list_path
        self.sigma = sigma

        self.crop_size = crop_size
        self.max_iters = max_iters

        self.iter = iter_start

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) * batch_size / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            gtpointlab_file = osp.join(self.root_gtpointlab, name)
            partialab_file = osp.join(self.root_partialab, name)
            det_file = osp.join(self.root_det, name)
            self.files.append({
                "img": img_file,
                "gtpointlab": gtpointlab_file,
                "partialab": partialab_file,
                "det": det_file,
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

        gtpointlab1 = Image.open(datafiles1["gtpointlab"])  #0,1
        gtpointlab1 = np.asarray(gtpointlab1, np.float32)
        gtpointlab2 = Image.open(datafiles2["gtpointlab"])
        gtpointlab2 = np.asarray(gtpointlab2, np.float32)

        flabel1 = Image.open(datafiles1["partialab"])  #0,1,255
        flabel1 = np.asarray(flabel1, np.float32)
        flabel2 = Image.open(datafiles2["partialab"])
        flabel2 = np.asarray(flabel2, np.float32)
        
        det1 = Image.open(datafiles1["det"])  #0-255
        det1 = np.asarray(det1, np.float32)
        det2 = Image.open(datafiles2["det"])
        det2 = np.asarray(det2, np.float32)
        
        lab_list1 = [gtpointlab1, flabel1, det1]
        if np.random.random() > 0.1:
            lab_list2 = [gtpointlab2, flabel2, det2]
            image1, lab_list1 = cut_and_paste_v1(image1, lab_list1, image2, lab_list2, cut_size=(256, 256))

        image1 = min_max(image1, max=1, min=0)
        image_as_np, flabe1_as_np, gtpointlab_as_np, det_as_np = aug_target_img_lab3(image1, lab_list1[1], lab_list1[0], lab_list1[2], self.crop_size)
        image_as_np = min_max(image_as_np, max=1, min=0)
       
        size = image_as_np.shape

        image_as_np = np.expand_dims(image_as_np, axis=0)
        image_as_tensor = torch.from_numpy(image_as_np.astype("float32")).float()
        
        gtpoint = generate_center_point(gtpointlab_as_np.astype('uint8'))
        gtpoints_as_tensor = torch.from_numpy(gtpoint.astype("float32")).float()

        flabel_as_tensor = torch.from_numpy(flabe1_as_np.astype("float32")).long()

        flabe1_as_np1 = flabe1_as_np.copy()
        flabe1_as_np1[flabe1_as_np1 != 1] = 0
        fpoints_map_np, fdetection_map_np = generate_center_gaussianmap(flabe1_as_np1.astype('uint8'), sigma=(self.sigma, self.sigma))

        fpoints_as_tensor = torch.from_numpy(fpoints_map_np.astype("float32")).float()
        fdetection_map_np = np.expand_dims(fdetection_map_np, axis=0)
        fdetection_map_as_tensor = torch.from_numpy(fdetection_map_np.astype("float32")).float()
        
        detbg_as_np = np.zeros_like(det_as_np)
        detbg_as_np[det_as_np == 0] = 1
    
        return image_as_tensor, gtpoints_as_tensor, flabel_as_tensor, \
                fpoints_as_tensor, fdetection_map_as_tensor, detbg_as_np, np.array(size)


class targetDataSet_train_step1_full(data.Dataset):
    def __init__(self, root_img, root_gtpoint, list_path, max_iters=None, iter_start=0, crop_size=[512, 512],
                  sigma=61, batch_size=1):

        self.root_img = root_img
        self.root_gtpoint = root_gtpoint
        self.list_path = list_path
        self.sigma = sigma

        self.crop_size = crop_size
        self.max_iters = max_iters

        self.iter = iter_start

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) * batch_size / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            gtpoint_file = osp.join(self.root_gtpoint, name)
          
            self.files.append({
                "img": img_file,
                "gtpoint": gtpoint_file,
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

        gtpoint1 = Image.open(datafiles1["gtpoint"]) 
        gtpoint1 = np.asarray(gtpoint1, np.float32)
        gtpoint2 = Image.open(datafiles2["gtpoint"])
        gtpoint2 = np.asarray(gtpoint2, np.float32)
        
        gtpoint1[gtpoint1 > 0] = 1
        gtpoint2[gtpoint2 > 0] = 1
       
        if np.random.random() > 0.5:
            image1, gtpoint1 = cut_and_paste(image1, gtpoint1, image2, gtpoint2, cut_size=(256, 256))

        image1 = min_max(image1, max=1, min=0)
        image_as_np, gtpoint_as_np = aug_target_img_lab(image1, gtpoint1, self.crop_size)
        image_as_np = min_max(image_as_np, max=1, min=0)
       
        size = image_as_np.shape

        image_as_np = np.expand_dims(image_as_np, axis=0)
        image_as_tensor = torch.from_numpy(image_as_np.astype("float32")).float()
        
        gtpoints_as_tensor = torch.from_numpy(gtpoint_as_np.astype("float32")).float()

        detection_map_np = make_gaussian_map(gtpoint_as_np.astype('uint8'), sigma=(self.sigma, self.sigma))
        detection_map_np = np.expand_dims(detection_map_np, axis=0)
        detection_map_as_tensor = torch.from_numpy(detection_map_np.astype("float32")).float()
    
        return image_as_tensor, gtpoints_as_tensor, detection_map_as_tensor, np.array(size)


class targetDataSet_train_step2_full(data.Dataset):
    def __init__(self, root_img, root_gtpointlab, list_path, max_iters=None, iter_start=0, crop_size=[512, 512],
                  sigma=61, batch_size=1):

        self.root_img = root_img
        self.root_gtpointlab = root_gtpointlab
        self.list_path = list_path
        self.sigma = sigma

        self.crop_size = crop_size
        self.max_iters = max_iters

        self.iter = iter_start

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) * batch_size / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            gtpointlab_file = osp.join(self.root_gtpointlab, name)
            self.files.append({
                "img": img_file,
                "gtpointlab": gtpointlab_file,
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

        gtpointlab1 = Image.open(datafiles1["gtpointlab"])  #0,1,255
        gtpointlab1 = np.asarray(gtpointlab1, np.float32)
        gtpointlab2 = Image.open(datafiles2["gtpointlab"])
        gtpointlab2 = np.asarray(gtpointlab2, np.float32)

        if np.random.random() > 0.5:
            image1, gtpointlab1 = cut_and_paste(image1, gtpointlab1, image2, gtpointlab2, cut_size=(256, 256))

        image1 = min_max(image1, max=1, min=0)
        image_as_np, gtpointlab_as_np = aug_target_img_lab(image1, gtpointlab1, self.crop_size)
        image_as_np = min_max(image_as_np, max=1, min=0)
       
        size = image_as_np.shape

        image_as_np = np.expand_dims(image_as_np, axis=0)
        image_as_tensor = torch.from_numpy(image_as_np.astype("float32")).float()
      
        flabel_as_tensor = torch.from_numpy(gtpointlab_as_np.astype("float32")).long()

        flabe1_as_np1 = gtpointlab_as_np.copy()
        flabe1_as_np1[flabe1_as_np1 != 1] = 0
        points_map_np, detection_map_np = generate_center_gaussianmap(flabe1_as_np1.astype('uint8'), sigma=(self.sigma, self.sigma))

        points_as_tensor = torch.from_numpy(points_map_np.astype("float32")).float()
        detection_map_np = np.expand_dims(detection_map_np, axis=0)
        detection_map_as_tensor = torch.from_numpy(detection_map_np.astype("float32")).float()
    
        return image_as_tensor, flabel_as_tensor, points_as_tensor, detection_map_as_tensor, np.array(size)



class targetDataSet_val(data.Dataset):
    def __init__(self, root_img, root_label, list_path, max_iters=None, batch_size=1):

        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) * batch_size / len(self.img_ids)))
        self.files = []

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
        datafiles = self.files[index]
        name = datafiles["name"]

        image = Image.open(datafiles["img"])
        label = Image.open(datafiles["label"])

        image_as_np = np.asarray(image, np.float32)
        label_as_np = np.asarray(label, np.float32)
    
        original_label = torch.from_numpy(np.asarray(label_as_np) / 255)
        original_label[original_label > 0] = 1

        image_as_np = min_max(image_as_np, max=1, min=0)
        image_as_tensor = torch.Tensor(image_as_np)
       
        return image_as_tensor, original_label, name


class counting_testdatset(data.Dataset):
    def __init__(self, root_img, root_label, list_path, ignore_path=None, max_iters=None, sigma=61, batch_size=1):

        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path

        self.sigma = sigma

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) * batch_size / len(self.img_ids)))
        self.files = []

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
        datafiles = self.files[index]
        name = datafiles["name"]

        image = Image.open(datafiles["img"])
        label = Image.open(datafiles["label"])

        image_as_np = np.asarray(image, np.float32)
        label_as_np = np.asarray(label, np.float32)

        image_as_np = min_max(image_as_np, max=1, min=0)
        label_as_np[label_as_np > 0] = 1

        label_as_np = morphology.remove_small_objects(label_as_np.astype('bool'), min_size=4, connectivity=1).astype('uint8')
        gt_num, _ = cv2.connectedComponents(label_as_np.astype('uint8'))
        gt_num_tensor = torch.from_numpy(np.asarray((gt_num - 1))).long()

        points_map_np, detmap_np = generate_center_gaussianmap(label_as_np, sigma=(self.sigma, self.sigma))
        image_as_tensor = torch.tensor(image_as_np)
        detmap_tensor = torch.from_numpy(detmap_np)

        label_as_np = torch.from_numpy(label_as_np).long()

        return image_as_tensor, gt_num_tensor, detmap_tensor, name

