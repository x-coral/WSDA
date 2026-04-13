import cv2
import numpy as np
import os

img_dir = '././data/mixR/ori_sparse_lab0/'
save_dir = '././data/mixR/ori_sparse_lab/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# size = 1536
# step = 1536


# img_list = os.listdir(img_dir)
# for file_name in sorted(img_list):
#     img_path = os.path.join(img_dir, file_name)
#     img = cv2.imread(img_path, -1)
#     print(file_name)
#     # 计算需要分割的行数和列数
#     rows = int(np.ceil((img.shape[0] - size) / step)) + 1
#     cols = int(np.ceil((img.shape[1] - size) / step)) + 1
#     # 循环分割图像
#     for i in range(rows):
#         for j in range(cols):
#             # 计算切片的左上角坐标和右下角坐标
#             x1 = j * step
#             y1 = i * step
#             x2 = min(x1 + size, img.shape[1])
#             y2 = min(y1 + size, img.shape[0])

#             # 计算切片的保存名称
#             save_name = os.path.join(save_dir, f"{file_name[:-4]}_{i}_{j}.png")

#             # 提取切片图像并保存
#             im = img[y1:y2, x1:x2]
#             cv2.imwrite(save_name, im)


# # 不重叠分割
# img_list = os.listdir(img_dir)
# for file_name in sorted(img_list):
#     img_path = os.path.join(img_dir, file_name)
#     img = cv2.imread(img_path, -1)
#     # if file_name != 'im0360.png':
#     #     continue
#     print(file_name)

#     for i in range(2):
#         for j in range(2):
#             save_name = os.path.join(save_dir, file_name.replace(file_name[-4:], "") + "_" + str(j * 4 + i) + file_name[-4:])
#             im = img[i * 2048:i * 2048 + 2048, j * 2048:j * 2048 + 2048]
#             cv2.imwrite(save_name, im)


def fenge(path, path_out, size_w=1024, size_h=1024, step=512): #重叠度为1024-512
    ims_list=os.listdir(path)
    count = 0
    for im_list in ims_list:
        number = 0
        name = im_list[:-4]  #去处“.png后缀”
        print(name)
        img = cv2.imread(path+im_list, -1)
        size = img.shape
        if size[0]>=size_h and size[1]>=size_w:
            count = count + 1
            for h in range(0,size[0]-step-1,step):
                star_h = h
                for w in range(0,size[1]-step-1,step):
                    star_w = w
                    end_h = star_h + size_h
                    if end_h > size[0]:
                        star_h = size[0] - size_h
                        end_h = star_h + size_h
                    end_w = star_w + size_w
                    if end_w > size[1]:
                        star_w = size[1] - size_w
                    end_w = star_w + size_w
                    cropped = img[star_h:end_h, star_w:end_w]
                    name_img = name + '_' + str(star_h) + '_' + str(star_w) 
                    #name_img = name + '_' + str(number)
                    cv2.imwrite('{}/{}.png'.format(path_out, name_img), cropped)
                    number = number + 1

        print('图片{}切割成{}张'.format(name, number))

    print('共完成{}张图片'.format(count))

#
#
# def pinjie(path, path_out, size_w=4096, size_h=4096, step=512): #重叠度为512
#     ims_list = os.listdir(path)
#     ims_list.sort()
#     count = 1
#     w = step
#     h = step
#     newdata = np.zeros((size_w, size_h))
#
#     for im_list in ims_list:
#         print(im_list)
#         name = im_list[:6] + '.png'
#         aa = im_list[7:-4]
#         i = int(aa[:aa.index("_")])
#         j = int(aa[aa.index("_") + 1:])
#         img = cv2.imread(path + im_list, -1)
#         size = img.shape
#         data = np.array(img)
#         newdata[i:i + 1024, j:j + 1024] += data
#         if count % 49 == 0:
#             newdata[512:4096-512, 512:4096-512] *= 0.5
#             newdata[newdata>0]=255
#             #print(newdata)
#             newimg = np.array(newdata).astype(np.uint8)
#             cv2.imwrite('{}/{}'.format(path_out, name), newimg)
#             print(name+'拼接成功')
#             newdata = np.zeros((size_w, size_h))
#         count +=1
#
# #
if __name__ == '__main__':
    #pinjie(img_dir, save_dir, size_w=4096, size_h=4096, step=512)
    fenge(img_dir, save_dir, size_w=1024, size_h=1024, step=512)
    #fenge(img_dir, save_dir, size_w=1600, size_h=1600, step=1248)


