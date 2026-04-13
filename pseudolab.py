

import os
import torch
import cv2
import tqdm
from torch.utils import data
from skimage import morphology
import torch.nn.functional as F

from model.HSC82 import CoDA_Net
from tool.tools_self import *
from utils.stats_utils import *
from dataset.target_dataset import targetDataSet_val
# from tool.tools_self import standar_gaussian
from val import save_det_pred

def select_sparse_plabel(sparse_point_path, predictlabel_path, save_path):
    #只筛选有点标注且预测出来的实例

    make_dirs(save_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(predictlabel_path))):

        plabel_path = os.path.join(predictlabel_path, file_name)
        pointmap_path = os.path.join(sparse_point_path, file_name)

        predict_map = cv2.imread(plabel_path, -1).astype('uint8')
        sparse_point_map = cv2.imread(pointmap_path, -1).astype('uint8')

        # if len(np.unique(predict_map)) >=3:
        predict_map[predict_map != 1] = 0

        sparse_point_map[sparse_point_map > 0] = 1
        _, connections = cv2.connectedComponents(predict_map)
        unique = np.unique(connections * sparse_point_map)[1:]

        pred_sparse_map = np.in1d(connections, unique).reshape(predict_map.shape).astype('uint8')
        
        save_filename = os.path.join(save_path, file_name)
        cv2.imwrite(save_filename, pred_sparse_map*255)
        

def generate_sparse_plabel(sparse_point_path, predictlabel_path, save_path):
    
    #筛选有点标注且预测出来的实例 + 点标注的邻域实例 

    make_dirs(save_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(predictlabel_path))):

        plabel_path = os.path.join(predictlabel_path, file_name)
        pointmap_path = os.path.join(sparse_point_path, file_name)

        predict_map = cv2.imread(plabel_path, -1).astype('uint8')
        sparse_point_map = cv2.imread(pointmap_path, -1).astype('uint8')

        # if len(np.unique(predict_map)) >=3:
        predict_map[predict_map != 1] = 0

        sparse_point_map[sparse_point_map > 0] = 1
        _, connections = cv2.connectedComponents(predict_map)
        unique = np.unique(connections * sparse_point_map)[1:]

        pred_sparse_map = np.in1d(connections, unique).reshape(predict_map.shape).astype('uint8')

        unsparse_point_map = sparse_point_map.copy()
        unsparse_point_map[(unsparse_point_map>0) &(pred_sparse_map>0)] = 0
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 13))
        point_dilate = cv2.dilate(unsparse_point_map, kernel, iterations=3)
        
        add_map = cv2.bitwise_or(pred_sparse_map, point_dilate)

        save_filename = os.path.join(save_path, file_name)
        cv2.imwrite(save_filename, add_map)  #0,1



def generate_sparse_plabel_v1(sparse_point_path, predictlabel_path, save_path):
    
    #筛选有点标注且预测出来的实例 + 点标注的邻域实例 + 邻域周围设为忽略

    make_dirs(save_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(predictlabel_path))):

        plabel_path = os.path.join(predictlabel_path, file_name)
        pointmap_path = os.path.join(sparse_point_path, file_name)

        predict_map0 = cv2.imread(plabel_path, -1).astype('uint8')
        sparse_point_map = cv2.imread(pointmap_path, -1).astype('uint8')

        predict_map = predict_map0.copy()
        # if len(np.unique(predict_map)) >=3:
        predict_map[predict_map != 1] = 0

        sparse_point_map[sparse_point_map > 0] = 1
        _, connections = cv2.connectedComponents(predict_map)
        unique_connected = np.unique(connections * sparse_point_map)[1:]
        missing_connected = np.setdiff1d(np.unique(connections), unique_connected)[1:]

        pred_sparse_map = np.in1d(connections, unique_connected).reshape(predict_map.shape).astype('uint8')  #tp
        pred_fp_map = np.in1d(connections, missing_connected).reshape(predict_map.shape).astype('uint8')  #fp

        unsparse_point_map = sparse_point_map.copy()
        unsparse_point_map[(unsparse_point_map>0) &(pred_sparse_map>0)] = 0
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 13))
        point_dilate = cv2.dilate(unsparse_point_map, kernel, iterations=3)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 33))
        point_dilate1 = cv2.dilate(unsparse_point_map, kernel1, iterations=3)
     
        add_map = cv2.bitwise_or(pred_sparse_map, point_dilate)  #foreground
        add_map[add_map!=1] = 255    #ignore
        add_map[(add_map == 255) & (predict_map0 == 0)] = 0   #background
        add_map[(add_map != 1) & (pred_fp_map > 0)] = 0   #background
        add_map[(add_map == 0) & (point_dilate1>0)] = 255  #邻域周围设为忽略

        save_filename = os.path.join(save_path, file_name)  

        cv2.imwrite(save_filename, add_map)  #0,1,255
        
        
def select_slabel(sparse_point_map, predict_map):

    sparse_point_map[sparse_point_map > 0] = 1
    _, connections = cv2.connectedComponents(predict_map)
    unique = np.unique(connections * sparse_point_map)[1:]

    pred_sparse_map = np.in1d(connections, unique).reshape(predict_map.shape).astype('uint8')

    return  pred_sparse_map

 
def generate_slabel(sparse_point_map, predict_map):

    sparse_point_map[sparse_point_map > 0] = 1
    _, connections = cv2.connectedComponents(predict_map)
    unique = np.unique(connections * sparse_point_map)[1:]

    pred_sparse_map = np.in1d(connections, unique).reshape(predict_map.shape).astype('uint8')
    
    unsparse_point_map = sparse_point_map.copy()
    unsparse_point_map[(unsparse_point_map>0) &(pred_sparse_map>0)] = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 13))
    point_dilate = cv2.dilate(unsparse_point_map, kernel, iterations=3)
    point_dilate[point_dilate > 0] = 1
    
    add_map = cv2.bitwise_or(pred_sparse_map, point_dilate)

    return  add_map


def generate_slabel_background(fpoint_map, sparse_point_map, binary_map, predict_map):

    fpoint_map[fpoint_map > 0] =1
    sparse_point_map[sparse_point_map > 0] = 1
    
    _, connections = cv2.connectedComponents(binary_map)
    unique = np.unique(connections * fpoint_map)[1:]

    pred_sparse_map = np.in1d(connections, unique).reshape(binary_map.shape).astype('uint8')
    
    unsparse_point_map = sparse_point_map.copy()
    unsparse_point_map[(unsparse_point_map>0) &(pred_sparse_map>0)] = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 13))
    point_dilate = cv2.dilate(unsparse_point_map, kernel, iterations=3)
    point_dilate[point_dilate > 0] = 1
    
    add_map = cv2.bitwise_or(pred_sparse_map, point_dilate)
    
    add_map_bg = add_map.copy()
    add_map_bg[add_map_bg != 1] = 255
    add_map_bg[(add_map_bg == 255) & (predict_map == 0)] = 0  #background
    
    kernel2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 33))
    point_dilate2 = cv2.dilate(unsparse_point_map, kernel2, iterations=3)
    
    add_map_bg[(add_map_bg != 1) & (point_dilate2 > 0)] = 255    #邻域周围设为忽略
    
    return  add_map, add_map_bg



#!!! 
def peak_local_maxima_v1(input, gt_partiallab, num, win_size, ratio):

    input = np.expand_dims(input, axis=0)
    input = np.expand_dims(input, axis=0)
    input_as_tensor = torch.from_numpy(input.astype('float32')).float().cuda()
    # input_as_tensor[input_as_tensor < torch.max(input_as_tensor * 0.5)] = 0
    offset = (win_size - 1) // 2
    padding = torch.nn.ConstantPad2d(offset,float("-inf"))
    padded_maps = padding(input_as_tensor)
    batch_size, num_channels,h,w = padded_maps.size()
    element_map = torch.arange(0,h*w).long().view(1,1,h,w)[:,:,offset:-offset,offset:-offset].float().cuda()

    _, indices = F.max_pool2d(padded_maps, kernel_size = win_size,stride=1,return_indices = True)
    peak_map = (indices == element_map)

    peak = (peak_map.data[0, 0, :] + 0).cpu().numpy()
    # num = np.count_nonzero(peak)
    final_num = int(np.ceil(num * ratio))
    
    peak_filter = peak * gt_partiallab
    peak[peak_filter != 0] = 0

    peak = np.expand_dims(peak, axis=0)
    peak = np.expand_dims(peak, axis=0)
    peak_map = torch.from_numpy(peak>0).cuda()

    crm = input_as_tensor * peak_map.float()

    for i in range(final_num):
        indices = torch.argmax(crm)
        crm[:, :, torch.div(indices, crm.size()[3], rounding_mode='trunc'), indices % crm.size()[3]] = float("-Inf")
       
    mask1 = crm == float('-Inf')
    mask1 = (mask1 & peak_map)
    mask_array = (mask1.data[0, 0, :] + 0).cpu().numpy()

    return  mask_array


def peak_local_maxima_v2(input, num, win_size, ratio0, ratio):

    input = np.expand_dims(input, axis=0)
    input = np.expand_dims(input, axis=0)
    input_as_tensor = torch.from_numpy(input.astype('float32')).float().cuda()
    # input_as_tensor[input_as_tensor < torch.max(input_as_tensor * 0.5)] = 0
    offset = (win_size - 1) // 2
    padding = torch.nn.ConstantPad2d(offset,float("-inf"))
    padded_maps = padding(input_as_tensor)
    batch_size, num_channels,h,w = padded_maps.size()
    element_map = torch.arange(0,h*w).long().view(1,1,h,w)[:,:,offset:-offset,offset:-offset].float().cuda()

    _, indices = F.max_pool2d(padded_maps, kernel_size = win_size,stride=1,return_indices = True)
    peak_map = (indices == element_map)

    peak = (peak_map.data[0, 0, :] + 0).cpu().numpy()
    # num = np.count_nonzero(peak)
    final_num0 = int(np.ceil(num * ratio0))
    final_num = int(np.ceil(num * ratio))

    peak = np.expand_dims(peak, axis=0)
    peak = np.expand_dims(peak, axis=0)
    peak_map = torch.from_numpy(peak>0).cuda()

    crm0 = input_as_tensor * peak_map.float()
    crm = input_as_tensor * peak_map.float()

    for i in range(final_num):
        indices = torch.argmax(crm)
        crm[:, :, torch.div(indices, crm.size()[3], rounding_mode='trunc'), indices % crm.size()[3]] = float("-Inf")
        if i < final_num0:
            crm0[:, :, torch.div(indices, crm0.size()[3], rounding_mode='trunc'), indices % crm0.size()[3]] = float("-Inf")
       
    mask0 = crm0 == float('-Inf')
    mask0 = (mask0 & peak_map)
    mask_array0 = (mask0.data[0, 0, :] + 0).cpu().numpy()
    
    mask1 = crm == float('-Inf')
    mask1 = (mask1 & peak_map)
    mask_array = (mask1.data[0, 0, :] + 0).cpu().numpy()

    return  mask_array0, mask_array

        
        
#!!!
def from_detectionmap_generate_pseudolab_v1(gtpoint_path, segmap_path, detectionmap_path, save_lab_path, save_point_path, save_pselab_path, ratio, point_num):
    make_dirs(save_lab_path)
    make_dirs(save_pselab_path)
    make_dirs(save_point_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(detectionmap_path))):
        
        postprocess_labmap_save_path = os.path.join(save_lab_path, file_name)
        postprocess_pselabmap_save_path = os.path.join(save_pselab_path, file_name)
        postprocess_pointmap_save_path = os.path.join(save_point_path, file_name)
        
        det_path = os.path.join(detectionmap_path, file_name)
        pred_lab_path = os.path.join(segmap_path, file_name)
        gtpoint_pth = os.path.join(gtpoint_path, file_name)

        detection_map = cv2.imread(det_path, -1)
        predict_map = cv2.imread(pred_lab_path, -1)
        gtpoint = cv2.imread(gtpoint_pth, -1)
        gtpoint[gtpoint>0]=1
        
        binary_predict_map =  predict_map.copy()
        binary_predict_map[binary_predict_map != 1] = 0

        binary_predict_map = morphology.remove_small_objects(binary_predict_map.astype('bool'), min_size=4, connectivity=1)
        binary_predict_map = binary_predict_map.astype('uint8')
        
        gtpoint_partiallab = select_slabel(gtpoint, binary_predict_map)  # 0,1

        fpoints_map = peak_local_maxima_v1(detection_map, gtpoint_partiallab, point_num[file_name], win_size=3, ratio=ratio)

        fpoints_map = cv2.bitwise_or(fpoints_map.astype('uint8'), gtpoint)
        cv2.imwrite(postprocess_pointmap_save_path, fpoints_map)
        
        final_predict_lab, final_pseudo_lab = generate_slabel_background(fpoints_map, gtpoint, binary_predict_map, predict_map)  
        cv2.imwrite(postprocess_labmap_save_path, final_predict_lab)    # 0,1
        cv2.imwrite(postprocess_pselabmap_save_path, final_pseudo_lab)  # 0,1,255
  
  
  
def from_detectionmap_generate_pseudolab_v2(segmap_path, detectionmap_path, save_maxlab_path, ratio0, save_lab_path, save_point_path, save_pselab_path, ratio, point_num):
    make_dirs(save_maxlab_path)
    make_dirs(save_lab_path)
    make_dirs(save_pselab_path)
    make_dirs(save_point_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(detectionmap_path))):
        
        postprocess_maxlabmap_save_path = os.path.join(save_maxlab_path, file_name)
        postprocess_labmap_save_path = os.path.join(save_lab_path, file_name)
        postprocess_pselabmap_save_path = os.path.join(save_pselab_path, file_name)
        postprocess_pointmap_save_path = os.path.join(save_point_path, file_name)
        
        det_path = os.path.join(detectionmap_path, file_name)
        pred_lab_path = os.path.join(segmap_path, file_name)
       
        detection_map = cv2.imread(det_path, -1)
        predict_map = cv2.imread(pred_lab_path, -1)
        
        binary_predict_map =  predict_map.copy()
        binary_predict_map[binary_predict_map != 1] = 0

        binary_predict_map = morphology.remove_small_objects(binary_predict_map.astype('bool'), min_size=4, connectivity=1)
        binary_predict_map = binary_predict_map.astype('uint8')

        fpoints_map0, fpoints_map = peak_local_maxima_v2(detection_map, point_num[file_name], win_size=3, ratio0=ratio0, ratio=ratio)
        cv2.imwrite(postprocess_pointmap_save_path, fpoints_map)

        maxpoint_partiallab = select_slabel(fpoints_map0, binary_predict_map)  # 0,1
        cv2.imwrite(postprocess_maxlabmap_save_path, maxpoint_partiallab)
        
        final_predict_lab = generate_slabel(fpoints_map.astype('uint8'), binary_predict_map)  # 0,1
        cv2.imwrite(postprocess_labmap_save_path, final_predict_lab)
        
        final_pseudo_lab = final_predict_lab.copy()  # 0,1,255
        final_pseudo_lab[final_pseudo_lab != 1] = 255
        final_pseudo_lab[(final_pseudo_lab == 255) & (predict_map == 0)] = 0  #background
        cv2.imwrite(postprocess_pselabmap_save_path, final_pseudo_lab)
    
  
        
#!!!
def generate_det_background(sparse_point_path, pseudodet_save_path, save_path, sigma, thre=0):

    make_dirs(save_path)
    for file_name in tqdm.tqdm(sorted(os.listdir(pseudodet_save_path))):
        
        point_path = os.path.join(sparse_point_path, file_name)
        plabel_path = os.path.join(pseudodet_save_path, file_name)
        
        sparse_point = cv2.imread(point_path, -1).astype('uint8')
        predict_map = cv2.imread(plabel_path, -1).astype('uint8')
        
        sparse_point[sparse_point > 0] = 1  #gtpoint
        
        predict_map[predict_map < predict_map.max()*thre] = 0
        
        background = np.zeros_like(predict_map)
        background[predict_map == 0] = 1  
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sigma, sigma))
        point_dilate = cv2.dilate(sparse_point, kernel, iterations=1)
       
        background[point_dilate > 0] = 0  
        
        save_filename = os.path.join(save_path, file_name)

        cv2.imwrite(save_filename, background) #0,1


def generate_pseudolabel_threshold(model, valloader, save_dir, det_save_path, gpu, usecuda, sigma, threshold):
    if usecuda:
        model.cuda(gpu)
    model.eval()
    make_dirs(save_dir)
    make_dirs(det_save_path)
    cofficient = standar_gaussian(sigma)
    point_num={}
    for i_pic, (images_v, original_msk, name) in tqdm.tqdm(enumerate(valloader)):

        with torch.no_grad():
            if usecuda:
                image_v = images_v.unsqueeze(1).cuda(gpu)
            else:
                image_v = images_v.unsqueeze(1)
            try:
                seg_output, det_output, _= model(image_v)
                doutput = torch.zeros_like(det_output)
                output = torch.nn.functional.softmax(seg_output, dim=1)
                doutput += det_output

                seg_output, det_output, _= model(image_v.flip(dims=(2,)))
                seg_output = seg_output.flip(dims=(2,))
                output += torch.nn.functional.softmax(seg_output, dim=1)
                doutput += det_output.flip(dims=(2,))

                seg_output, det_output, _ = model(image_v.flip(dims=(3,)))
                seg_output = seg_output.flip(dims=(3,))
                output += torch.nn.functional.softmax(seg_output, dim=1)
                doutput += det_output.flip(dims=(3,))

                seg_output, det_output, _= model(image_v.flip(dims=(2, 3)))
                seg_output = seg_output.flip(dims=(2, 3))
                output += torch.nn.functional.softmax(seg_output, dim=1)
                doutput += det_output.flip(dims=(2, 3))

                output = output / 4.0
                doutput = doutput / 4.0

                output = output[:, 1, :, :].float().cpu()

                seg_output = output[:, (images_v.shape[-2] - original_msk.shape[-2]):,
                             (images_v.shape[-1] - original_msk.shape[-1]):]
                det_output = doutput[:, :, (images_v.shape[-2] - original_msk.shape[-2]):,
                             (images_v.shape[-1] - original_msk.shape[-1]):]
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
            
            for i in range(seg_output.shape[0]):
                img_cont_np = seg_output[i,:,:].cpu().numpy()
                label = np.ones_like(img_cont_np) * 255
                label[img_cont_np > threshold] = 1
                label[img_cont_np < (1-threshold)] = 0

                save_name = os.path.join(save_dir, str(name[i]))
                cv2.imwrite(save_name, label)

                save_det_pred((det_output[i, 0, :, :]), name[i], det_save_path)
                densitymap = det_output[i, 0, :, :].clone()
                num = np.round((torch.sum(densitymap) / cofficient).cpu().numpy())
                point_num[name[i]] = num
                print(name[i], num)
    return point_num


def val_threshold(test_model_path, save_path, det_save_path, args, threshold=0.5):
    setup_seed(20)
    batch_size = 4

    makedatalist(args.data_dir_target, args.data_list_target)
    # args.data_list_target = 'dataset/MitoEMR_list/parttrain1.txt'

    model = CoDA_Net(in_channels=1, out_channels=2, device="cuda:" + str(args.gpu))
    model.load_state_dict(torch.load(test_model_path, map_location="cuda:" + str(args.gpu)),strict=False)

    testloader = data.DataLoader(
        targetDataSet_val(args.data_dir_target, args.data_dir_target_label, args.data_list_target),
    batch_size=batch_size, shuffle=False)

    point_num = generate_pseudolabel_threshold(model, testloader, save_path, det_save_path, args.gpu, True, args.sigma, threshold=threshold)

    return point_num
