import cv2
import numpy as np
import os
import shutil
import sys
import torch
import random

def standar_gaussian(kernel_size):
    gt = np.zeros((kernel_size, kernel_size))
    gt[int(kernel_size/2)][int(kernel_size/2)] = 1
    count_map = cv2.GaussianBlur(gt, (kernel_size, kernel_size), 0, borderType=0)
    am = np.amax(count_map)
    return 255/am

def makedatalist(imgpath, listpath):
    num = 0
    file_path = imgpath
    path_list = os.listdir(file_path)
    for file_name in path_list:
        path = imgpath + file_name+'/'
        if os.path.isdir(path):
            num = num + 1
            path_list.extend(os.listdir(path))
        else:
            break

    del path_list[:num]
    path_list.sort()

    with open(listpath, 'a') as f:
        f.seek(0)
        f.truncate()
        for file_name in path_list:
            f.write(file_name + '\n')

    f.close()
    
def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        # print('create dir:',dir)
        
def remove_or_create_exp_dir(exp_name):
    print(exp_name)
    if os.path.exists(exp_name):
        wait = input("do you want to rm the dir,'y'or'n'\n")
        if wait == 'y':
            print('rm the exp dir\n')
            shutil.rmtree(exp_name)
        else:
            make_dirs(exp_name)
        
def bak_code(code_path_list,exp_name):
    for path in code_path_list:
        make_dirs(exp_name + '/code_bak/')
        shutil.copyfile(path, exp_name + '/code_bak/'+path.split('/')[-1])
        print('bak change code in the path:',exp_name + '/code_bak/'+path.split('/')[-1])

class Logger(object):
    def __init__(self, filename='logprocess.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    # print(worker_id)
    GLOBAL_WORKER_ID = worker_id
    setup_seed(worker_id + 20)
    

def polarize(img):
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img    


if __name__ ==  "__main__":
    pass