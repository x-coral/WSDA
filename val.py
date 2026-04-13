import os
import torch
import cv2
import tqdm
from model.HSC82 import CoDA_Net
from tool.tools_self import *
from utils.stats_utils import *
from tool.tools_self import standar_gaussian

def save_seg_pred(input, im_name, save_folder_name, jac):
    # img_cont_np = input.data[0].cpu().numpy()
    img_cont_np = input.data.cpu().numpy()
    img_cont_np = polarize(img_cont_np)

    img_cont = (img_cont_np * 255).astype('uint8')

    save_name = os.path.join(save_folder_name, str(im_name))
    text = str('%.3f' % jac)
    cv2.putText(img_cont, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    cv2.imwrite(save_name, img_cont)

    return img_cont_np

def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x

def save_det_pred(input, im_name, save_folder_name):
    img_cont_np = input.data.cpu().numpy()
    img_cont_np = MaxMinNormalization(img_cont_np, np.max(img_cont_np), np.min(img_cont_np))

    img_cont = (img_cont_np * 255).astype('uint8')

    save_name = os.path.join(save_folder_name, str(im_name))
    cv2.imwrite(save_name, img_cont)

    return img_cont_np



def validate_model(model, valloader, seg_save_dir, det_save_dir,  gpu, usecuda, type='kplus'):

    model.cuda(gpu)
    model.eval()
    total_dice = 0
    total_jac = 0
    count = 0

    seg = torch.zeros((4096, 4096))
    det = torch.zeros((4096, 4096))
    mask = torch.zeros((4096, 4096))

    for i_pic, (images_v, original_msk, name) in tqdm.tqdm(enumerate(valloader)):

        with torch.no_grad():
            if usecuda:
                image_v = images_v.unsqueeze(1).cuda(gpu)
            else:
                image_v = images_v.unsqueeze(1)
            try:
                seg_output, det_output, _ = model(image_v)
                seg_output = torch.softmax(seg_output, dim=1)[:, 1, :, :].float().cpu()

                seg_output = seg_output[:, (images_v.shape[-2] - original_msk.shape[-2]):,
                             (images_v.shape[-1] - original_msk.shape[-1]):]
                det_output = det_output[:, :, (images_v.shape[-2] - original_msk.shape[-2]):,
                             (images_v.shape[-1] - original_msk.shape[-1]):]
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

        if type != 'mito':
            for i in range(seg_output.shape[0]):
                seg_cpu = seg_output[i,:,:].cpu().numpy()
                seg_cpu = polarize(seg_cpu)
                dice, jac = dice_coeff(seg_cpu, original_msk)

                save_seg_pred(seg_output[i,:,:], name[i], seg_save_dir, jac)

                save_det_pred((det_output[i, 0, :, :]), name[i], det_save_dir)

                total_dice = total_dice + dice
                total_jac = total_jac + jac

                count = count + 1

        else:
            for i in range(seg_output.shape[0]):
                
                newname = (''.join(name[i]))[:6]+'.png'
                
                if i_pic!=0 and oldname != newname:
                    seg[512:3584, :] /= 2
                    seg[:, 512:3584] /= 2

                    det[512:3584, :] /= 2
                    det[:, 512:3584] /= 2

                    mask[512:3584, :] /= 2
                    mask[:, 512:3584] /= 2
                    mask = mask.cpu().numpy()
                    mask[mask > 0] = 1

                    seg_cpu = seg.cpu().numpy()
                    seg_cpu = polarize(seg_cpu)
                    dice, jac = dice_coeff(seg_cpu, mask)

                    save_seg_pred(seg, oldname, seg_save_dir, jac)
                    save_det_pred(det, oldname, det_save_dir)

                    total_dice = total_dice + dice
                    total_jac = total_jac + jac

                    count = count + 1
                    seg = torch.zeros((4096, 4096))
                    det = torch.zeros((4096, 4096))
                    mask = torch.zeros((4096, 4096))
                    
                fromx = int(name[i].split("_")[1])
                endx = fromx + 1024
                fromy = int(name[i].split("_")[2].replace(".png", ""))
                endy = fromy + 1024

                seg[fromx:endx, fromy:endy] += seg_output[i, :, :]
                det[fromx:endx, fromy:endy] += det_output[i, 0, :, :].cpu()
                mask[fromx:endx, fromy:endy] += original_msk[i, :, :]
                
                oldname = newname


    return total_dice / count, total_jac / count



def valcount_model(model, valloader, sigma, gpu, usecuda):
    if usecuda:
        model.cuda(gpu)
    model.eval()
    count = 0
    total_error = 0
    total_mse = 0
    precount = []
    gt_count = []
    
    cofficient = standar_gaussian(sigma)
    
    for i_pic, (images_v, gt_num, detmap, name) in enumerate(valloader):
        with torch.no_grad():
            if usecuda:
                image_v = images_v.unsqueeze(1).cuda(gpu)
            else:
                image_v = images_v.unsqueeze(1)
            try:
                _, det_output,_ = model(image_v)
                new_output = det_output[:, :, (images_v.shape[-2] - detmap.shape[-2]):,
                             (images_v.shape[-1] - detmap.shape[-1]):]

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
                

        new_output_normalized = 255*(new_output - new_output.min()) / (new_output.max() - new_output.min()+1e-8)
        new_detmap_normalized = 255*(detmap - detmap.min()) / (detmap.max() - detmap.min()+1e-8)
        mse = np.mean(np.square(new_output_normalized.data[0][0].cpu().numpy().astype(np.float32) 
                                - new_detmap_normalized.data[0].cpu().numpy().astype(np.float32)))
        total_mse = total_mse + mse

        su = torch.sum(new_output, dim=(2, 3)) / cofficient
        count_one = np.round(su.data.cpu().numpy()[0][0])
        gt = gt_num.cpu().numpy()
        error = np.abs(gt - count_one)
        total_error = total_error + error
        count = count+1
        gt_count.append(gt)
        precount.append(count_one)
        # print(name[0], 'predict:', count_one, 'GT:', gt[0],  'mae:', error[0], 'mse:',mse)

    return total_mse/count, total_error/count