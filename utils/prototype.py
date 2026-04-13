import torch
import torch.nn.functional as F
import math
import cv2
from sklearn.cluster import KMeans
import numpy as np

def generate_queue(queue, sample, num):
    if len(queue) + len(sample) > num:
        # 计算需要弹出的元素数量
        num_to_pop = len(queue) + len(sample) - num
        # 弹出需要的数量元素
        if num_to_pop>num:
            num_to_pop=num
        for _ in range(min(num_to_pop, len(queue))):
            queue.popleft()
    queue.extend(sample)
    return queue

def generate_source_proto_down(f, predict, label, obj_queue_positive, gpu, threshold):
    centroid_0_obj_matrix_positive = []
    original_msk_tensor = label.unsqueeze(1).cuda(gpu)
    predict[:, 1:2, :, :][predict[:, 1:2, :, :] > threshold] = 1
    predict[:, 1:2, :, :][predict[:, 1:2, :, :] <= threshold] = 0
    predict[:, 0:1, :, :][predict[:, 0:1, :, :] > threshold] = 1
    predict[:, 0:1, :, :][predict[:, 0:1, :, :] <= threshold] = 0

    predict[:, 1:2, :, :] = predict[:, 1:2, :, :] * original_msk_tensor
    predict[:, 0:1, :, :] = predict[:, 0:1, :, :] * (1 - original_msk_tensor)

    # 下采样预测结果 predict
    predict_downsampled = F.interpolate(predict, size=f.size()[2:], mode='nearest')
    # predict_downsampled = F.interpolate(predict, size=f.size()[2:], mode='bilinear')
    # predict_downsampled[predict_downsampled > 0] = 1
    mask = predict_downsampled[:, 1:2, :, :]
    
    feature_0_obj = f.cuda(gpu) * mask

    # all_instance_average prototype
    target_0_obj_cnt1 = torch.sum(mask, dim=[0, 2, 3], keepdim=True)
    if target_0_obj_cnt1 != 0:
        centroid_0_obj1 = torch.sum(feature_0_obj, dim=[0, 2, 3], keepdim=True)
        centroid_0_obj1 /= target_0_obj_cnt1
        centroid_0_obj_matrix_positive.append(centroid_0_obj1.cuda(gpu))
       
    if len(centroid_0_obj_matrix_positive) :
        obj_queue_positive = generate_queue(queue=obj_queue_positive, sample=centroid_0_obj_matrix_positive, num=1)
        
    return obj_queue_positive


def generate_source_proto_up(f, predict, label, obj_queue_positive, gpu, threshold):
    centroid_0_obj_matrix_positive = []
    original_msk_tensor = label.unsqueeze(1).cuda(gpu)
    predict[:, 1:2, :, :][predict[:, 1:2, :, :] > threshold] = 1
    predict[:, 1:2, :, :][predict[:, 1:2, :, :] <= threshold] = 0
    predict[:, 0:1, :, :][predict[:, 0:1, :, :] > threshold] = 1
    predict[:, 0:1, :, :][predict[:, 0:1, :, :] <= threshold] = 0

    predict[:, 1:2, :, :] = predict[:, 1:2, :, :] * original_msk_tensor
    predict[:, 0:1, :, :] = predict[:, 0:1, :, :] * (1 - original_msk_tensor)

    mask = predict[:, 1:2, :, :]
    # 上采样
    feature = F.interpolate(f.cuda(gpu), size=mask.size()[2:], mode='bilinear')
    feature_0_obj = feature * mask

    # all_instance_average prototype
    target_0_obj_cnt1 = torch.sum(mask, dim=[0, 2, 3], keepdim=True)
    if target_0_obj_cnt1 != 0:
        centroid_0_obj1 = torch.sum(feature_0_obj, dim=[0, 2, 3], keepdim=True)
        centroid_0_obj1 /= target_0_obj_cnt1
        centroid_0_obj_matrix_positive.append(centroid_0_obj1.cuda(gpu))
       
    if len(centroid_0_obj_matrix_positive) :
        obj_queue_positive = generate_queue(queue=obj_queue_positive, sample=centroid_0_obj_matrix_positive, num=1)
        
    return obj_queue_positive


def generate_random_anchor_stage1_thre_down(feature, predict, threshold1, threshold2, gpu):
    
    feature = feature.cuda(gpu)
    
    A = torch.zeros_like(predict).cuda(gpu)  #easy
    A[predict > threshold1] = 1

    B = torch.zeros_like(predict).cuda(gpu)  #hard
    B[(threshold2 < predict) & (predict < threshold1)] = 1
    
    # 使用插值函数对label进行下采样
    A_downsampled_label = F.interpolate(A, size=feature.size()[2:], mode='nearest')
    B_downsampled_label = F.interpolate(B, size=feature.size()[2:], mode='nearest')
    
    mask = A_downsampled_label == 1  # 创建布尔掩码
    mask = mask.expand(-1, feature.shape[1], -1, -1)
    A_extracted_features = feature[mask].view(-1, feature.shape[1])
    N, C = A_extracted_features.shape
    K = min(math.floor(N*0.50), 512)
    random_indices = torch.randperm(N)[:K]
    A_anchor = A_extracted_features[random_indices]
    
    mask = B_downsampled_label == 1  # 创建布尔掩码
    mask = mask.expand(-1, feature.shape[1], -1, -1)
    B_extracted_features = feature[mask].view(-1, feature.shape[1])
    N, C = B_extracted_features.shape
    K = min(math.floor(N*0.50), 512)
    random_indices = torch.randperm(N)[:K]
    B_anchor = B_extracted_features[random_indices]
    
    if A_anchor.nelement() == 0:
        combined_anchor = B_anchor
    elif B_anchor.nelement() == 0:
        combined_anchor = A_anchor
    else:
        combined_anchor = torch.cat((A_anchor, B_anchor), dim=0)
        
    return combined_anchor


def generate_random_anchor_stage1_thre_up(feature, predict, threshold1, threshold2, gpu):
    
    feature = feature.cuda(gpu)
    
    A = torch.zeros_like(predict).cuda(gpu)  #easy
    A[predict > threshold1] = 1

    B = torch.zeros_like(predict).cuda(gpu)  #hard
    B[(threshold2 < predict) & (predict < threshold1)] = 1
    
    feature = F.interpolate(feature, size=predict.size()[2:], mode='bilinear')
    
    mask = A == 1  # 创建布尔掩码
    mask = mask.expand(-1, feature.shape[1], -1, -1)
    A_extracted_features = feature[mask].view(-1, feature.shape[1])
    N, C = A_extracted_features.shape
    K = min(math.floor(N*0.50), 512)
    random_indices = torch.randperm(N)[:K]
    A_anchor = A_extracted_features[random_indices]
    
    mask = B == 1  # 创建布尔掩码
    mask = mask.expand(-1, feature.shape[1], -1, -1)
    B_extracted_features = feature[mask].view(-1, feature.shape[1])
    N, C = B_extracted_features.shape
    K = min(math.floor(N*0.50), 512)
    random_indices = torch.randperm(N)[:K]
    B_anchor = B_extracted_features[random_indices]
    
    if A_anchor.nelement() == 0:
        combined_anchor = B_anchor
    elif B_anchor.nelement() == 0:
        combined_anchor = A_anchor
    else:
        combined_anchor = torch.cat((A_anchor, B_anchor), dim=0)
        
    return combined_anchor


def generate_random_anchor_stage1_sort_down(feature, predict, threshold, gpu):
    
    anchors = []
    
    batch_size = feature.size(0)  # 获取批次大小
    feature = feature.cuda(gpu)
    
    predicted_class = (predict > threshold).byte()

    for batch_idx in range(batch_size):
        predicted_class_batch = predicted_class[batch_idx]
        predict_batch = predict[batch_idx]

        # 获取预测正确的像素的索引
        correct_indices = torch.nonzero(predicted_class_batch.view(-1), as_tuple=False).squeeze()

        # 按预测概率降序排列
        sorted_indices = predict_batch.view(-1).argsort(descending=True)

        num = min(math.floor(torch.numel(correct_indices) * 0.50), 512)
        if num > 0:
            # 随机取预测正确概率前1000个中500个的
            bottom_200_correct_indices = sorted_indices[correct_indices][-(2 * num):]
            random_indices = torch.randperm(bottom_200_correct_indices.shape[0])[:num]
            bottom_100_correct_indices = bottom_200_correct_indices[random_indices]
            
            # 随机取预测正确中500个的
            random_indices = torch.randperm(sorted_indices[correct_indices][:-(2 * num)].shape[0])[:num]
            random_50_correct_indices = sorted_indices[correct_indices][random_indices]
            
            # 合并两个部分
            final_indices = torch.cat([random_50_correct_indices, bottom_100_correct_indices])

            # 创建一个 0 的 tensor，将对应索引处的值置为 1
            result_tensor = torch.zeros(1, 1, predict.shape[-2], predict.shape[-1]).cuda(gpu)
            result_tensor.view(-1)[final_indices] = 1

            # 使用插值函数对label进行下采样
            downsampled_label = F.interpolate(result_tensor, size=feature.size()[2:], mode='nearest')
        
            mask_indices = (downsampled_label == 1).nonzero(as_tuple=False)[:, 2:4]  # 提取非零元素的索引
            extracted_vectors = feature[batch_idx, :, mask_indices[:, 0], mask_indices[:, 1]]   # 根据位置索引从feature中提取对应的向量
            
            # 将提取的向量整理成（num_anchors，feature_size）的形状
            anchor = extracted_vectors.reshape(-1, feature.shape[1])

            # 将每个batch的anchor存储起来
            anchors.append(anchor)
        else:
            # 如果没有符合条件的点，则返回空张量
            anchors.append(torch.tensor([]).cuda(gpu))

    # 将所有批次的anchor合并成一个张量
    anchors = torch.stack(anchors, dim=0)  # 如果某个批次没有有效的anchor，则会返回空张量

    return anchors


def generate_random_anchor_stage1_sort_up(feature, predict, threshold, gpu):
    
    anchors = []
    
    batch_size = feature.size(0)  # 获取批次大小
    feature = feature.cuda(gpu)
    feature = F.interpolate(feature, size=predict.size()[2:], mode='bilinear')
    
    predicted_class = (predict > threshold).byte()

    for batch_idx in range(batch_size):
        predicted_class_batch = predicted_class[batch_idx]
        predict_batch = predict[batch_idx]

        # 获取预测正确的像素的索引
        correct_indices = torch.nonzero(predicted_class_batch.view(-1), as_tuple=False).squeeze()

        # 按预测概率降序排列
        sorted_indices = predict_batch.view(-1).argsort(descending=True)

        num = min(math.floor(torch.numel(correct_indices) * 0.50), 512)
        if num > 0:
            # 随机取预测正确概率前1000个中500个的
            bottom_200_correct_indices = sorted_indices[correct_indices][-(2 * num):]
            random_indices = torch.randperm(bottom_200_correct_indices.shape[0])[:num]
            bottom_100_correct_indices = bottom_200_correct_indices[random_indices]
            
            # 随机取预测正确中500个的
            random_indices = torch.randperm(sorted_indices[correct_indices][:-(2 * num)].shape[0])[:num]
            random_50_correct_indices = sorted_indices[correct_indices][random_indices]
            
            # 合并两个部分
            final_indices = torch.cat([random_50_correct_indices, bottom_100_correct_indices])

            # 创建一个 0 的 tensor，将对应索引处的值置为 1
            result_tensor = torch.zeros(1, 1, predict.shape[-2], predict.shape[-1]).cuda(gpu)
            result_tensor.view(-1)[final_indices] = 1
            mask_indices = (result_tensor == 1).nonzero(as_tuple=False)[:, 2:4]  # 提取非零元素的索引
            extracted_vectors = feature[batch_idx, :, mask_indices[:, 0], mask_indices[:, 1]]   # 根据位置索引从feature中提取对应的向量
            
            # 将提取的向量整理成（num_anchors，feature_size）的形状
            anchor = extracted_vectors.reshape(-1, feature.shape[1])

            # 将每个batch的anchor存储起来
            anchors.append(anchor)
        else:
            # 如果没有符合条件的点，则返回空张量
            anchors.append(torch.tensor([]).cuda(gpu))

    # 将所有批次的anchor合并成一个张量
    anchors = torch.cat(anchors, dim=0)  # 如果某个批次没有有效的anchor，则会返回空张量

    return anchors





def generate_random_anchor_stage2_sort_up(feature, predict, flabel, gpu):
    
    anchors = []
    
    batch_size = feature.size(0)  # 获取批次大小
    feature = feature.cuda(gpu)
    feature = F.interpolate(feature, size=predict.size()[2:], mode='bilinear')
    
    flabel = flabel.unsqueeze(0).cuda(gpu)
    
    predicted_class = (flabel == 1).byte()

    for batch_idx in range(batch_size):
        predicted_class_batch = predicted_class[batch_idx]
        predict_batch = predict[batch_idx]

        # 获取预测正确的像素的索引
        correct_indices = torch.nonzero(predicted_class_batch.view(-1), as_tuple=False).squeeze()

        # 按预测概率降序排列
        sorted_indices = predict_batch.view(-1).argsort(descending=True)

        num = min(math.floor(torch.numel(correct_indices) * 0.50), 512)
        if num > 0:
            # 随机取预测正确概率前1000个中500个的
            bottom_200_correct_indices = sorted_indices[correct_indices][-(2 * num):]
            random_indices = torch.randperm(bottom_200_correct_indices.shape[0])[:num]
            bottom_100_correct_indices = bottom_200_correct_indices[random_indices]
            
            # 随机取预测正确中500个的
            random_indices = torch.randperm(sorted_indices[correct_indices][:-(2 * num)].shape[0])[:num]
            random_50_correct_indices = sorted_indices[correct_indices][random_indices]
            
            # 合并两个部分
            final_indices = torch.cat([random_50_correct_indices, bottom_100_correct_indices])

            # 创建一个 0 的 tensor，将对应索引处的值置为 1
            result_tensor = torch.zeros(1, 1, predict.shape[-2], predict.shape[-1]).cuda(gpu)
            result_tensor.view(-1)[final_indices] = 1
            mask_indices = (result_tensor == 1).nonzero(as_tuple=False)[:, 2:4]  # 提取非零元素的索引
            extracted_vectors = feature[batch_idx, :, mask_indices[:, 0], mask_indices[:, 1]]   # 根据位置索引从feature中提取对应的向量
            
            # 将提取的向量整理成（num_anchors，feature_size）的形状
            anchor = extracted_vectors.reshape(-1, feature.shape[1])

            # 将每个batch的anchor存储起来
            anchors.append(anchor)
        else:
            # 如果没有符合条件的点，则返回空张量
            anchors.append(torch.tensor([]).cuda(gpu))

    # 将所有批次的anchor合并成一个张量
    anchors = torch.cat(anchors, dim=0)  # 如果某个批次没有有效的anchor，则会返回空张量

    return anchors












def generate_target_proto_stage1_down(f, predict, point, obj_queue_positive, bck_queue_positive, obj_queue_negative, bck_queue_negative, gpu, threshold):
    centroid_0_bck_matrix_positive = []
    centroid_0_bck_matrix_negative = []
    centroid_0_obj_matrix_positive = []
    centroid_0_obj_matrix_negative = []
    predict[:, 1:2, :, :][predict[:, 1:2, :, :] > threshold] = 1
    predict[:, 1:2, :, :][predict[:, 1:2, :, :] <= threshold] = 0
    predict[:, 0:1, :, :][predict[:, 0:1, :, :] > threshold] = 1
    predict[:, 0:1, :, :][predict[:, 0:1, :, :] <= threshold] = 0

    # ignore pixel
    predict[:, 1:2, :, :][predict[:, 1:2, :, :] > 1] = 0
    predict[:, 0:1, :, :][predict[:, 0:1, :, :] < 0] = 0

    # 下采样预测结果 predict
    predict_downsampled = F.interpolate(predict, size=f.size()[2:], mode='nearest')

    mask = predict_downsampled[:, 1:2, :, :]
    mask_bck = predict_downsampled[:, 0:1, :, :]

    feature = f.cuda(gpu)
   
    # sparse prototype
    pred_sparse_map = torch.zeros_like(predict[:, 1:2, :, :])
    
    for idx in range(predict.shape[0]):
        predict_map = predict[:, 1:2, :, :][idx, 0, :, :].cpu().numpy()
        predict_map = cv2.convertScaleAbs(predict_map)
        _, connections = cv2.connectedComponents(predict_map)
        unique = np.unique(connections * point[idx,:,:].float().cpu().numpy())[1:]
        sparse_map = np.in1d(connections, unique).reshape(predict_map.shape).astype('uint8')
        pred_sparse_map[idx:idx+1, 0:1,:,:] = torch.from_numpy(sparse_map).cuda(gpu)
        

    if pred_sparse_map.sum() != 0:
        # 下采样预测结果 pred_sparse_map
        pred_sparse_map_downsampled = F.interpolate(pred_sparse_map, size=f.size()[2:], mode='nearest')
        selected_features = feature * pred_sparse_map_downsampled.cuda(gpu)
        # 对特征张量进行平均池化操作，得到（1，128，1，1）的向量
        pooled_features = torch.sum(selected_features, dim=[0, 2, 3], keepdim=True) / pred_sparse_map_downsampled.sum()
        centroid_0_obj_matrix_positive.append(pooled_features)
        centroid_0_bck_matrix_negative.append(pooled_features)

            
    # all_instance_average prototype
    target_0_obj_cnt1 = torch.sum(mask, dim=[0, 2, 3], keepdim=True)
    if target_0_obj_cnt1 != 0:
        centroid_0_obj1 = torch.sum(feature * mask, dim=[0, 2, 3], keepdim=True)
        centroid_0_obj1 /= target_0_obj_cnt1
        centroid_0_obj_matrix_positive.append(centroid_0_obj1.cuda(gpu))
        centroid_0_bck_matrix_negative.append(centroid_0_obj1.cuda(gpu))

    #cluster
    obj_matrix = feature * mask
    obj_matrix = obj_matrix.permute(0, 2, 3, 1)
    obj_matrix = torch.flatten(obj_matrix, start_dim=0, end_dim=2)
    non_zero_indices = torch.any(obj_matrix != 0, dim=1)
    obj_matrix = obj_matrix[non_zero_indices]

    if obj_matrix.shape[0] > 1:
        km_obj = KMeans(n_clusters=2, n_init=10, init='k-means++').fit(obj_matrix.detach().cpu().numpy())
        centroids_obj = km_obj.cluster_centers_
        centroid_0_bck_matrix_negative.append((torch.tensor(centroids_obj[0]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))
        centroid_0_bck_matrix_negative.append((torch.tensor(centroids_obj[1]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))

    
    # all_background_average prototype
    target_0_bck_cnt = torch.sum(mask_bck, dim=[0, 2, 3], keepdim=True)
    if target_0_bck_cnt != 0:
        centroid_0_bck = torch.sum(feature * mask_bck, dim=[0, 2, 3], keepdim=True)
        centroid_0_bck /= target_0_bck_cnt
        centroid_0_bck_matrix_positive.append(centroid_0_bck)
        centroid_0_obj_matrix_negative.append(centroid_0_bck)

    #cluster
    bck_matrix = feature * mask_bck
    bck_matrix = bck_matrix.permute(0, 2, 3, 1)
    bck_matrix = torch.flatten(bck_matrix, start_dim=0, end_dim=2)
    non_zero_indices = torch.any(bck_matrix != 0, dim=1)
    bck_matrix = bck_matrix[non_zero_indices]
  
    if bck_matrix.shape[0] > 1:
        km_bck = KMeans(n_clusters=2, n_init=10, init='k-means++').fit(bck_matrix.detach().cpu().numpy())
        centroids_bck = km_bck.cluster_centers_
        centroid_0_obj_matrix_negative.append((torch.tensor(centroids_bck[0]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))
        centroid_0_obj_matrix_negative.append((torch.tensor(centroids_bck[1]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))
     

    if len(centroid_0_obj_matrix_positive) and len(centroid_0_bck_matrix_positive):
        obj_queue_positive = generate_queue(queue=obj_queue_positive, sample=centroid_0_obj_matrix_positive, num=2)
        obj_queue_negative = generate_queue(queue=obj_queue_negative, sample=centroid_0_obj_matrix_negative, num=3)
        bck_queue_positive = generate_queue(queue=bck_queue_positive, sample=centroid_0_bck_matrix_positive, num=1)
        bck_queue_negative = generate_queue(queue=bck_queue_negative, sample=centroid_0_bck_matrix_negative, num=4)
        

    return obj_queue_positive, bck_queue_positive, obj_queue_negative, bck_queue_negative


def generate_target_proto_stage1_up(f, predict, point, obj_queue_positive, bck_queue_positive, obj_queue_negative, bck_queue_negative, gpu, threshold):
    centroid_0_bck_matrix_positive = []
    centroid_0_bck_matrix_negative = []
    centroid_0_obj_matrix_positive = []
    centroid_0_obj_matrix_negative = []
    predict[:, 1:2, :, :][predict[:, 1:2, :, :] > threshold] = 1
    predict[:, 1:2, :, :][predict[:, 1:2, :, :] <= threshold] = 0
    predict[:, 0:1, :, :][predict[:, 0:1, :, :] > threshold] = 1
    predict[:, 0:1, :, :][predict[:, 0:1, :, :] <= threshold] = 0

    # ignore pixel
    predict[:, 1:2, :, :][predict[:, 1:2, :, :] > 1] = 0
    predict[:, 0:1, :, :][predict[:, 0:1, :, :] < 0] = 0

    mask = predict[:, 1:2, :, :]
    mask_bck = predict[:, 0:1, :, :]

    feature = F.interpolate(f.cuda(gpu), size=predict.size()[2:], mode='bilinear')
   
    # sparse prototype
    pred_sparse_map = torch.zeros_like(predict[:, 1:2, :, :])
    
    for idx in range(predict.shape[0]):
        predict_map = predict[:, 1:2, :, :][idx, 0, :, :].cpu().numpy()
        predict_map = cv2.convertScaleAbs(predict_map)
        _, connections = cv2.connectedComponents(predict_map)
        unique = np.unique(connections * point[idx,:,:].float().cpu().numpy())[1:]
        sparse_map = np.in1d(connections, unique).reshape(predict_map.shape).astype('uint8')
        pred_sparse_map[idx:idx+1, 0:1,:,:] = torch.from_numpy(sparse_map).cuda(gpu)
        

    if pred_sparse_map.sum() != 0:
        selected_features = feature * pred_sparse_map.cuda(gpu)
        # 对特征张量进行平均池化操作，得到（1，128，1，1）的向量
        pooled_features = torch.sum(selected_features, dim=[0, 2, 3], keepdim=True) / pred_sparse_map.sum()
        centroid_0_obj_matrix_positive.append(pooled_features)
        centroid_0_bck_matrix_negative.append(pooled_features)

            
    # all_instance_average prototype
    target_0_obj_cnt1 = torch.sum(mask, dim=[0, 2, 3], keepdim=True)
    if target_0_obj_cnt1 != 0:
        centroid_0_obj1 = torch.sum(feature * mask, dim=[0, 2, 3], keepdim=True)
        centroid_0_obj1 /= target_0_obj_cnt1
        centroid_0_obj_matrix_positive.append(centroid_0_obj1.cuda(gpu))
        centroid_0_bck_matrix_negative.append(centroid_0_obj1.cuda(gpu))

    #cluster
    obj_matrix = feature * mask
    obj_matrix = obj_matrix.permute(0, 2, 3, 1)
    obj_matrix = torch.flatten(obj_matrix, start_dim=0, end_dim=2)
    non_zero_indices = torch.any(obj_matrix != 0, dim=1)
    obj_matrix = obj_matrix[non_zero_indices]

    if obj_matrix.shape[0] > 1:
        km_obj = KMeans(n_clusters=2, n_init=10, init='k-means++').fit(obj_matrix.detach().cpu().numpy())
        centroids_obj = km_obj.cluster_centers_
        centroid_0_bck_matrix_negative.append((torch.tensor(centroids_obj[0]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))
        centroid_0_bck_matrix_negative.append((torch.tensor(centroids_obj[1]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))

    
    # all_background_average prototype
    target_0_bck_cnt = torch.sum(mask_bck, dim=[0, 2, 3], keepdim=True)
    if target_0_bck_cnt != 0:
        centroid_0_bck = torch.sum(feature * mask_bck, dim=[0, 2, 3], keepdim=True)
        centroid_0_bck /= target_0_bck_cnt
        centroid_0_bck_matrix_positive.append(centroid_0_bck)
        centroid_0_obj_matrix_negative.append(centroid_0_bck)

    #cluster
    bck_matrix = feature * mask_bck
    bck_matrix = bck_matrix.permute(0, 2, 3, 1)
    bck_matrix = torch.flatten(bck_matrix, start_dim=0, end_dim=2)
    non_zero_indices = torch.any(bck_matrix != 0, dim=1)
    bck_matrix = bck_matrix[non_zero_indices]
  
    if bck_matrix.shape[0] > 1:
        km_bck = KMeans(n_clusters=2, n_init=10, init='k-means++').fit(bck_matrix.detach().cpu().numpy())
        centroids_bck = km_bck.cluster_centers_
        centroid_0_obj_matrix_negative.append((torch.tensor(centroids_bck[0]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))
        centroid_0_obj_matrix_negative.append((torch.tensor(centroids_bck[1]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))
     

    if len(centroid_0_obj_matrix_positive) and len(centroid_0_bck_matrix_positive):
        obj_queue_positive = generate_queue(queue=obj_queue_positive, sample=centroid_0_obj_matrix_positive, num=2)
        obj_queue_negative = generate_queue(queue=obj_queue_negative, sample=centroid_0_obj_matrix_negative, num=3)
        bck_queue_positive = generate_queue(queue=bck_queue_positive, sample=centroid_0_bck_matrix_positive, num=1)
        bck_queue_negative = generate_queue(queue=bck_queue_negative, sample=centroid_0_bck_matrix_negative, num=4)
        

    return obj_queue_positive, bck_queue_positive, obj_queue_negative, bck_queue_negative



def generate_target_proto_stage2_up(f, point, label, obj_queue_positive, bck_queue_positive, obj_queue_negative, bck_queue_negative, gpu):
    centroid_0_bck_matrix_positive = []
    centroid_0_bck_matrix_negative = []
    centroid_0_obj_matrix_positive = []
    centroid_0_obj_matrix_negative = []
    original_msk_tensor = label.unsqueeze(1).cuda(gpu)
    
    point[point > 0] = 1

    mask = original_msk_tensor
    mask_bck = 1 - original_msk_tensor

    #ignore pixel
    mask[mask > 1] = 0
    mask_bck[mask_bck < 0] = 0 
    
    feature = F.interpolate(f.cuda(gpu), size=original_msk_tensor.size()[2:], mode='bilinear', align_corners=True).cuda(gpu)
   
    # sparse prototype
    pred_sparse_map = torch.zeros_like(mask)
    
    for idx in range(mask.shape[0]):
        predict_map = mask[idx, 0, :, :].cpu().numpy()
        predict_map = cv2.convertScaleAbs(predict_map)
        _, connections = cv2.connectedComponents(predict_map)
        unique = np.unique(connections * point[idx,:,:].float().cpu().numpy())[1:]
        sparse_map = np.in1d(connections, unique).reshape(predict_map.shape).astype('uint8')
        pred_sparse_map[idx:idx+1, 0:1,:,:] = torch.from_numpy(sparse_map).cuda(gpu)
    

    if pred_sparse_map.sum() != 0:
        selected_features = feature * pred_sparse_map.cuda(gpu)
        # 对特征张量进行平均池化操作，得到（1，128，1，1）的向量
        pooled_features = torch.sum(selected_features, dim=[0, 2, 3], keepdim=True) / pred_sparse_map.sum()
        centroid_0_obj_matrix_positive.append(pooled_features)
        centroid_0_bck_matrix_negative.append(pooled_features)
    
    obj_matrix = feature * mask
    # all_instance_average prototype
    target_0_obj_cnt1 = torch.sum(mask, dim=[0, 2, 3], keepdim=True)
    if target_0_obj_cnt1 != 0:
        centroid_0_obj1 = torch.sum(obj_matrix, dim=[0, 2, 3], keepdim=True)
        centroid_0_obj1 /= target_0_obj_cnt1
        centroid_0_obj_matrix_positive.append(centroid_0_obj1.cuda(gpu))
        centroid_0_bck_matrix_negative.append(centroid_0_obj1.cuda(gpu))

    #cluster
    obj_matrix = obj_matrix.permute(0, 2, 3, 1)
    obj_matrix = torch.flatten(obj_matrix, start_dim=0, end_dim=2)
    non_zero_indices = torch.any(obj_matrix != 0, dim=1)
    obj_matrix = obj_matrix[non_zero_indices]

    if obj_matrix.shape[0] > 1:
        km_obj = KMeans(n_clusters=2, n_init=10, init='k-means++').fit(obj_matrix.detach().cpu().numpy())
        centroids_obj = km_obj.cluster_centers_
        centroid_0_bck_matrix_negative.append((torch.tensor(centroids_obj[0]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))
        centroid_0_bck_matrix_negative.append((torch.tensor(centroids_obj[1]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))

    # all_background_average prototype
    bck_matrix = feature * mask_bck
   
    target_0_bck_cnt = torch.sum(mask_bck, dim=[0, 2, 3], keepdim=True)
    if target_0_bck_cnt != 0:
        centroid_0_bck = torch.sum(bck_matrix, dim=[0, 2, 3], keepdim=True)
        centroid_0_bck /= target_0_bck_cnt
        centroid_0_bck_matrix_positive.append(centroid_0_bck)
        centroid_0_obj_matrix_negative.append(centroid_0_bck)
        
    bck_matrix = bck_matrix.permute(0, 2, 3, 1)
    bck_matrix = torch.flatten(bck_matrix, start_dim=0, end_dim=2)
    non_zero_indices = torch.any(bck_matrix != 0, dim=1)
    bck_matrix = bck_matrix[non_zero_indices]
    # random_indices = np.random.randint(0, bck_matrix.shape[0], size=int(0.02 * bck_matrix.shape[0]))

    if bck_matrix.shape[0] > 1:
        # bck_matrix = bck_matrix[random_indices]
        km_bck = KMeans(n_clusters=2, n_init=10, init='k-means++').fit(bck_matrix.detach().cpu().numpy())
        centroids_bck = km_bck.cluster_centers_
        centroid_0_obj_matrix_negative.append((torch.tensor(centroids_bck[0]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))
        centroid_0_obj_matrix_negative.append((torch.tensor(centroids_bck[1]).unsqueeze(1).unsqueeze(2).unsqueeze(0)).cuda(gpu))

    if len(centroid_0_obj_matrix_positive) and len(centroid_0_bck_matrix_positive):
        obj_queue_positive = generate_queue(queue=obj_queue_positive, sample=centroid_0_obj_matrix_positive, num=2)
        obj_queue_negative = generate_queue(queue=obj_queue_negative, sample=centroid_0_obj_matrix_negative, num=3)
        bck_queue_positive = generate_queue(queue=bck_queue_positive, sample=centroid_0_bck_matrix_positive, num=1)
        bck_queue_negative = generate_queue(queue=bck_queue_negative, sample=centroid_0_bck_matrix_negative, num=4)
        
    return obj_queue_positive, bck_queue_positive, obj_queue_negative, bck_queue_negative
