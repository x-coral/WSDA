from imports import *

def main(exp_dir, pseudoseg_save_path, pseudodet_save_path, ratio):
    
    # args.save_pred_every = 100

    print('============' + exp_dir + '============')

    partiallab_save_path = exp_dir + '/partiallab'  #0:seg_background  1:seg_foreground
    partialpnt_save_path = exp_dir + '/partialpnt'
    partial_pseudolabel_save_path = os.path.join(exp_dir, 'partial_pseudolabel') #0:seg_background  1:seg_foreground, 255:ignore

    gtpointlab_save_path = os.path.join(exp_dir, 'gtpoint_label')
    generate_sparse_plabel(args.data_dir_target_point, pseudoseg_save_path, gtpointlab_save_path)  # 0,1
    
    print('============From det Result Generate Partial Pseudo Label Start============')
    from_detectionmap_generate_pseudolab_v1(args.data_dir_target_point, pseudoseg_save_path, pseudodet_save_path, 
                                            partiallab_save_path, partialpnt_save_path,  partial_pseudolabel_save_path, ratio, point_num0)  
    print('============From det Result Generate Partial Pseudo Label End============')

    print('============mask_background============')
    det_background_save_path = os.path.join(exp_dir, 'det_background') 
    generate_det_background(args.data_dir_target_point, pseudodet_save_path, det_background_save_path, args.sigma, thre=0)

    makedatalist(args.data_dir_img, args.data_list)
    makedatalist(args.data_dir_target, args.data_list_target)
    makedatalist(args.data_dir_val, args.data_list_val)
    # args.data_list_val = './dataset/MitoEMH_list/val1.txt'

    trainloader = data.DataLoader(
        sourceDataSet_train(args.data_dir_img, args.data_dir_label, args.data_list,
                            max_iters=args.num_steps, crop_size=args.input_size, sigma=args.sigma, batch_size=args.batch_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    trainloader_iter = enumerate(trainloader)
        
    targetloader = data.DataLoader(
        targetDataSet_train_step1(args.data_dir_target, gtpointlab_save_path, partial_pseudolabel_save_path, det_background_save_path,
                                        args.data_list_target, max_iters=args.num_steps, iter_start=args.iter_start, 
                                        crop_size=args.input_size, sigma=args.sigma, batch_size=args.batch_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    targetloader_iter = enumerate(targetloader)

    valloader = data.DataLoader(
        targetDataSet_val(args.data_dir_val, args.data_dir_val_label, args.data_list_val,
                          batch_size=args.batch_size),
        batch_size=args.batch_size, shuffle=False)
        
    testloader = data.DataLoader(
        counting_testdatset(args.data_dir_target, args.data_dir_target_label,  './dataset/MitoEMR_list/parttrain11.txt',
                            sigma=args.sigma, batch_size=args.batch_size),
        batch_size=1, shuffle=False)
    
    print('load segmentation model from:', args.restore_from)
    model.load_state_dict(torch.load(args.restore_from, map_location="cuda:" + str(args.gpu)), strict=False)
    model.train()

    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=2000)

    sce = SCELoss(num_classes=2, a=1, b=0)
   
    obj_queue_source_positive = deque(maxlen=1)

    # obj_queue_target_positive = deque(maxlen=2)
    # obj_queue_target_negative = deque(maxlen=3)
    # bck_queue_target_positive = deque(maxlen=1)
    # bck_queue_target_negative = deque(maxlen=4)
    
    for i_iter in range(args.iter_start, args.num_steps+1):
    
        loss_seg_source_value = 0
        loss_seg_dice_source_value = 0
        loss_det_source_value = 0

        loss_seg_target_value = 0
        loss_seg_dice_target_value = 0
        loss_det_target_value = 0
        loss_contrast_target_value=0

        optimizer.zero_grad()

        _, batch = trainloader_iter.__next__()
        images, labels, points, gaussians, _, _ = batch
        if usecuda:
            images_source = images.cuda(args.gpu)
            points = points.cuda(args.gpu) 
        else:
            images_source = images

        seg_source, det_source, feature = model(images_source)

        # predict = torch.softmax(seg_source, dim=1)
        
        # if i_iter % 500 == 1:
        #      obj_queue_source_positive = generate_source_proto_up(feature.detach(), predict.detach(), labels,
        #                                               obj_queue_source_positive, args.gpu, threshold=0.95)
             
        loss_seg = sce(seg_source, labels, args.gpu)
        loss_seg_source_value += loss_seg.data.cpu().numpy()

        loss_det_source = weight_mse(det_source, gaussians, args.gpu)
        loss_det_source_value += loss_det_source.data.cpu().numpy()
       
        loss = loss_seg + 0.05 * loss_det_source 
        
        loss.backward()    

        _, batch = targetloader_iter.__next__()
        images, tlabels, tpoints, fpoints, fgaussians, detbackground, _ = batch
        if usecuda:
            images_target = images.cuda(args.gpu)
        else:
            images_target = images

        seg_target, det_target, feature = model(images_target)

        detbackground = detbackground.data.cpu().numpy()
        loss_det_target = weight_mse_partial_bg(det_target, detbackground, fpoints, fgaussians, args.gpu)

        loss_det_target_value += loss_det_target.data.cpu().numpy()

        # predict = torch.softmax(seg_target, dim=1)

        # target_obj_anchor = generate_random_anchor_stage1_sort_up(feature, predict[:, 1:2, :, :].detach(), 0.95, args.gpu)
        # target_bck_anchor = generate_random_anchor_stage1_sort_up(feature, predict[:, 0:1, :, :].detach(), 0.95, args.gpu)
       
        # if i_iter % 500 == 1:
        #      obj_queue_target_positive, bck_queue_target_positive, \
        #      obj_queue_target_negative, bck_queue_target_negative = \
        #          generate_target_proto_stage1_up(feature.detach(), predict.detach(), tpoints, 
        #                                               obj_queue_target_positive, bck_queue_target_positive,
        #                                               obj_queue_target_negative, bck_queue_target_negative, args.gpu, 0.95)

        # #将两个 deque 合并成一个列表，然后进行数乘操作
        # new_list = [x for x in list(obj_queue_target_positive)] + [x for x in list(obj_queue_source_positive)]
        # #将新的列表转换成 deque
        # obj_queue_positive = deque(new_list)
    
        # if len(obj_queue_positive) and len(bck_queue_target_positive):
        #     loss_contrast_target = prototype_contrast_loss(target_obj_anchor, target_bck_anchor, obj_queue_positive,
        #                                                    bck_queue_target_positive, obj_queue_target_negative,
        #                                                     bck_queue_target_negative, args.gpu, t=0.5)   #0.05
           
        #     loss_contrast_target_value += loss_contrast_target.item()
        # else:
        #     loss_contrast_target = 0
     
        loss = 0.01*loss_det_target
        # + 1*loss_contrast_target 
       
        loss.backward()
        optimizer.step()
    
        # if scheduler is not None:
        #     scheduler.step()
        #     args.learning_rate = scheduler.get_last_lr()[0]
       
        if (i_iter % 50 == 0):
            print(exp_dir.split('/')[-1] + '_time = {0},lr = {1: 5f}'.format(datetime.datetime.now(),
                                                   args.learning_rate))

            print('iter = {0:8d}/{1:8d}, loss_source_seg = {2:.5f} loss_seg_dice_source_value = {3:.5f}, loss_det_source_value = {4:.5f},'
                'loss_det_target = {5:.5f}, loss_seg_ce_target_value = {6:.5f}, loss_seg_dice_target_value= {7:.5f}, loss_contrast = {8:.5f} '
                .format(i_iter, args.num_steps, loss_seg_source_value, loss_seg_dice_source_value, loss_det_source_value,
                    loss_det_target_value, loss_seg_target_value, loss_seg_dice_target_value, loss_contrast_target_value))
            
        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            seg_val_dir = os.path.join(exp_dir, 'val/R', 'iter_' + str(i_iter), 'seg')
            det_val_dir = os.path.join(exp_dir, 'val/R', 'iter_' + str(i_iter), 'det')
            make_dirs(seg_val_dir)
            make_dirs(det_val_dir)

            dice, jac = validate_model(model, valloader, seg_val_dir, det_val_dir, args.gpu, usecuda, type='mito')
            print('val dice: %4f' % dice, 'val jac: %4f' % jac)
            if jac > args.best_tjac:
                args.best_tjac = jac
                print('best val dice: %4f' % dice, 'best val jac: %f' % jac)
                torch.save(model.state_dict(),
                          osp.join(args.snapshot_dir,'best.pth'))
                
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir, 'best'+ '_' +str(i_iter) + '.pth')) 
                
                val_detmap_mse, val_counting_error = valcount_model(model, testloader, args.sigma, args.gpu, usecuda)
                print('det_MSE: %3.2f' % val_detmap_mse, 'Count_MAE: %3.2f' % val_counting_error)
     
            if i_iter == args.num_steps:
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir, 'model'+str(i_iter) + '.pth'))

        scheduler.step(args.best_tjac)
        args.learning_rate = optimizer.param_groups[0]['lr']
                   

if __name__ == '__main__':

    args = get_arguments()

    exp_root_dir = './WSDA_cvlab2R/'
    exp_base_name = 'cvlab2R_15%_stage1_nocontrast_sgd'
    last_exp_dir =  "./pretrain_model/best_full_seg_contrast_det.pth"
    args.restore_from = "./pretrain_model/best_full_seg_contrast_det.pth"
   
    exp_base_dir = exp_root_dir + exp_base_name

    args.learning_rate = 0.00005
    args.gpu = 0

    code_path_list= ['./01_Stage1_WSDA_final.py', './add_arguments.py', './model/HSC82.py',
                     './dataset/data_aug.py', './dataset/source_dataset.py','./dataset/target_dataset.py', 
                     './utils/prototype.py', './val.py', './utils/loss.py', './pseudolab.py']
    bak_code(code_path_list,exp_base_dir)

    usecuda = True
    model = CoDA_Net(in_channels=1, out_channels=2, device=args.gpu)

    if usecuda:
        cudnn.benchmark = True
        cudnn.enabled = True
        model.cuda(args.gpu)
        
    for iter in range(0,5):

        args.num_steps = 10000
        # args.learning_rate = 0.00005
        exp_dir = exp_base_dir + '/iter' + str(iter)
        remove_or_create_exp_dir(exp_dir)

        args.snapshot_dir = exp_dir + '/snapshots'
        make_dirs(args.snapshot_dir)
        sys.stdout = Logger(stream=sys.stdout, filename=exp_dir + '/trainlog.log')
        sys.stderr = Logger(stream=sys.stderr, filename=exp_dir + '/trainlog.log')

        last_exp_model_dir = last_exp_dir + '/snapshots'
        last_model_path = os.path.join(last_exp_model_dir, 'best.pth')
        if os.path.exists(last_model_path):
            args.restore_from = last_model_path
            
        print('last_exp_model_dir: ', args.restore_from)

        print('============Generate Pseudo Label Start============')
        
        pseudoseg_save_path = exp_dir + '/pseudolab'
        pseudodet_save_path = exp_dir + '/pseudodet'
        # label_selected_ratio = max((iter+1) * 0.15-0.15, 0)
        label_selected_ratio = iter * 0.15

        point_num0 = val_threshold(args.restore_from, pseudoseg_save_path, pseudodet_save_path, args, threshold=0.9)
             
        print('============Generate Pseudo Label Start============')
        
        main(exp_dir, pseudoseg_save_path, pseudodet_save_path, label_selected_ratio)
        last_exp_dir = exp_dir
