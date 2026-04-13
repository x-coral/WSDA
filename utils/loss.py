import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2

import numpy as np

class SCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a #两个超参数
        self.b = b


    def forward(self, pred, labels, gpu):
        loss = F.cross_entropy(pred, labels.cuda(gpu), ignore_index=255)
       
        return loss
    
    
def weight_mse(pred_detmap, target_det, gpu):
    criterion = torch.nn.MSELoss().cuda(gpu)
    loss_det = criterion(pred_detmap, target_det.float().cuda(gpu))

    return loss_det


def weight_mse_partial_bg(pred_detmap, background, target_point, target_detmap, gpu):

    criterion = torch.nn.MSELoss(reduction='none').cuda(gpu)

    weight = torch.zeros(pred_detmap.shape).cuda(gpu)
    target_point = target_point.data.cpu().numpy()
    for i in range(weight.shape[0]):

        weight_map = cv2.GaussianBlur(target_point[i], (11, 11), 0, borderType=0)

        background_mask = (background[i] == True).astype("float32")
        mask = torch.from_numpy(background_mask).cuda(gpu)
        mask[target_detmap[i][0] > 0] = 1 

        if np.amax(weight_map) != 0:
            weight_map /= np.amax(weight_map)
        weight[i][0] = mask + 3 * torch.from_numpy(weight_map).cuda(gpu)
        # weight[i][0] = mask

    num_mask = weight > 0
    nums = torch.sum(num_mask)
    loss_count = torch.sum(weight * criterion(pred_detmap, target_detmap.float().cuda(gpu))) / (nums+1)

    return loss_count


def weight_mse_target_seg_adv(count_maps, count_semi, points, counts, gpu):

    criterion = torch.nn.MSELoss(reduction='none').cuda(gpu)

    weight_maps = torch.zeros(count_maps.shape).cuda(gpu)
    wpoints = points.data.cpu().numpy()
    for i in range(weight_maps.shape[0]):

        weight_map = cv2.GaussianBlur(wpoints[i], (11, 11), 0, borderType=0)

        background_mask = (count_semi[i] == True).astype("float32")
        mask = torch.from_numpy(background_mask).cuda(gpu)
        mask[counts[i][0] > 0] = 1 # 1

        if np.amax(weight_map) != 0:
            weight_map /= np.amax(weight_map)
        # weight_maps[i][0] = mask + 3 * torch.from_numpy(weight_map).cuda(gpu)
        weight_maps[i][0] = mask

    num_mask = weight_maps > 0
    nums = torch.sum(num_mask)
    loss_count = torch.sum(weight_maps * criterion(count_maps, counts.float().cuda(gpu))) / (nums+1)

    return loss_count

def prototype_contrast_loss(anchor_obj,anchor_bck,obj_set_positive,bck_set_positive,obj_set_negative,bck_set_negative,gpu, t=0.5):
    """
    Args:
        anchor_obj: K*128
        anchor_bck: K*128
        obj_set: 3*(1*128*1*1)
        bck_set: 3*(1*128*1*1)
    Returns:

    """
    if torch.numel(anchor_obj) == 0:
        obj_contrast_loss = torch.zeros(1).to(gpu)
        # print('skip_obj')
    else:
        anchor_obj = torch.nn.functional.normalize(anchor_obj, p=2, dim=1).requires_grad_()

        obj_set_positive = torch.stack(list(obj_set_positive)).squeeze(1).squeeze(-1).squeeze(-1).to(gpu)
        obj_set_negative = torch.stack(list(obj_set_negative)).squeeze(1).squeeze(-1).squeeze(-1).to(gpu)
        obj_set_positive = torch.nn.functional.normalize(obj_set_positive, p=2, dim=1).detach()
        obj_set_negative = torch.nn.functional.normalize(obj_set_negative, p=2, dim=1).detach()

        obj_set1 = torch.mean(obj_set_positive,dim=0).unsqueeze(0)
        obj_set2 = torch.nn.functional.normalize(obj_set1, p=2, dim=1)
        obj_obj_exp = torch.exp(torch.mm(anchor_obj, obj_set2.t())/t)

        size=obj_obj_exp.shape[1]
        obj_bck_exp = torch.exp(torch.mm(anchor_obj, obj_set_negative.t())/t)
        obj_bck_exp = obj_bck_exp.sum(dim=1, keepdim=True)
        obj_obj_exp = obj_obj_exp / (obj_bck_exp + obj_obj_exp+0.0001)
        obj_contrast_loss = -torch.mean(torch.log(obj_obj_exp+1e-5), dim=0)/size

    
    if torch.numel(anchor_bck) == 0:
        bck_contrast_loss = torch.zeros(1).to(gpu)
        # print('skip_bck')
    else:
        anchor_bck = torch.nn.functional.normalize(anchor_bck, p=2, dim=1).requires_grad_()

        bck_set_positive = torch.stack(list(bck_set_positive)).squeeze(1).squeeze(-1).squeeze(-1).to(gpu)
        bck_set_negative = torch.stack(list(bck_set_negative)).squeeze(1).squeeze(-1).squeeze(-1).to(gpu)
        bck_set_positive = torch.nn.functional.normalize(bck_set_positive, p=2, dim=1).detach()
        bck_set_negative = torch.nn.functional.normalize(bck_set_negative, p=2, dim=1).detach()

        bck_set1 = torch.mean(bck_set_positive, dim=0).unsqueeze(0)
        bck_set2 = torch.nn.functional.normalize(bck_set1, p=2, dim=1)
        bck_bck_exp = torch.exp(torch.mm(anchor_bck, bck_set2.t())/t)

        size=bck_bck_exp.shape[1]
        bck_obj_exp = torch.exp(torch.mm(anchor_bck, bck_set_negative.t())/t)
        bck_obj_exp = bck_obj_exp.sum(dim=1, keepdim=True)
        bck_bck_exp = bck_bck_exp / (bck_obj_exp + bck_bck_exp+0.0001)
        bck_contrast_loss = -torch.mean(torch.log(bck_bck_exp+1e-5), dim=0)/size

    contrast_loss=bck_contrast_loss+obj_contrast_loss
    return contrast_loss
