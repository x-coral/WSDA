from imports import *

def main(exp_dir, pseudoseg_save_path, pseudodet_save_path, ratio):

    print('============' + exp_dir + '============')

    partiallab_save_path = exp_dir + '/partiallab'  #0:seg_background  1:seg_foreground
    partialpnt_save_path = exp_dir + '/partialpnt'
    partial_pseudolabel_save_path = os.path.join(exp_dir, 'partial_pseudolabel') #0:seg_background  1:seg_foreground, 255:ignore

    gtpointlab_save_path = os.path.join(exp_dir, 'gtpoint_label')
    generate_sparse_plabel(args.data_dir_target_point, pseudoseg_save_path, gtpointlab_save_path)  # 0,1
    
    print('============From det Result Generate Partial Pseudo Label Start============')
    from_detectionmap_generate_pseudolab_v1(args.data_dir_target_point, pseudoseg_save_path, pseudodet_save_path, 
                                            partiallab_save_path, partialpnt_save_path,  partial_pseudolabel_save_path, ratio, point_num)  
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
        targetDataSet_train_step2(args.data_dir_target, gtpointlab_save_path, partial_pseudolabel_save_path, det_background_save_path,
                                        args.data_list_target, max_iters=args.num_steps, iter_start=args.iter_start, 
                                        crop_size=args.input_size, sigma=args.sigma, batch_size=args.batch_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    targetloader_iter = enumerate(targetloader)

    valloader = data.DataLoader(
        targetDataSet_val(args.data_dir_val, args.data_dir_val_label, args.data_list_val,
                          batch_size=args.batch_size),
        batch_size=args.batch_size, shuffle=False)


    print('load segmentation model from:', args.restore_from)
    model.load_state_dict(torch.load(args.restore_from, map_location="cuda:" + str(args.gpu)),strict=False)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.99))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.9)
    # optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9,0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=2000)

    sce = SCELoss(num_classes=2, a=1, b=0)

    # obj_queue_source_positive = deque(maxlen=1)
   
    # obj_queue_target_positive = deque(maxlen=2)
    # obj_queue_target_negative = deque(maxlen=3)
    # bck_queue_target_positive = deque(maxlen=1)
    # bck_queue_target_negative = deque(maxlen=4)
    
    for i_iter in range(args.iter_start, args.num_steps+1):

        loss_seg_target_value = 0
        loss_seg_dice_target_value = 0
        loss_det_target_value = 0
        loss_contrast_value = 0

        optimizer.zero_grad()

        _, batch = trainloader_iter.__next__()
        images, labels, _, _, _, _ = batch
        if usecuda:
            images_source = images.cuda(args.gpu)
        else:
            images_source = images

        seg_source, _, feature = model(images_source)

        # predict = torch.softmax(seg_source, dim=1)

        # if i_iter % 500 == 1:
        #      obj_queue_source_positive = generate_source_proto_up(feature.detach(), predict.detach(), labels,
        #                                               obj_queue_source_positive, args.gpu, threshold=0.95)
             
        _, batch = targetloader_iter.__next__()
        images, tpoints, flabels, fpoints, fgaussians, detbackground, _ = batch
        if usecuda:
            images_target = images.cuda(args.gpu)
        else:
            images_target = images

        seg_target, det_target, feature= model(images_target)

        loss_seg_target = sce(seg_target, flabels, args.gpu)
        loss_seg_target_value += loss_seg_target.data.cpu().numpy()

        detbackground = detbackground.data.cpu().numpy()
        loss_det_target = weight_mse_partial_bg(det_target, detbackground, fpoints, fgaussians, args.gpu)
        loss_det_target_value += loss_det_target.data.cpu().numpy()

        # predict = torch.softmax(seg_target, dim=1)
        
        # obj_anchor = generate_random_anchor_stage2_sort_up(feature, predict[:, 1:2, :, :].detach(), flabels, args.gpu)
        # bck_anchor = generate_random_anchor_stage2_sort_up(feature, predict[:, 0:1, :, :].detach(), 1-flabels, args.gpu)

        # if i_iter % 500 == 1:
        #     obj_queue_target_positive, bck_queue_target_positive, \
        #     obj_queue_target_negative, bck_queue_target_negative = \
        #         generate_target_proto_stage2_up(feature.detach(), tpoints, flabels, 
        #                                               obj_queue_target_positive,bck_queue_target_positive, 
        #                                               obj_queue_target_negative, bck_queue_target_negative, args.gpu)
            
        # #将两个 deque 合并成一个列表，然后进行数乘操作
        # new_list = [x for x in list(obj_queue_target_positive)] + [x for x in list(obj_queue_source_positive)]
        # #将新的列表转换成 deque
        # obj_queue_positive = deque(new_list)
    
        # if len(obj_queue_positive) and len(bck_queue_target_positive):
        #     loss_contrast_target = prototype_contrast_loss(obj_anchor, bck_anchor, obj_queue_positive,
        #                                                    bck_queue_target_positive, obj_queue_target_negative,
        #                                                     bck_queue_target_negative, args.gpu, t=0.5)   #0.05
           
        #     loss_contrast_value += loss_contrast_target.item()
        # else:
        #     loss_contrast_target = 0

        # loss = 10*loss_seg_target 
        loss = 0.01*loss_det_target + 1.0*loss_seg_target
        # + 0.005*loss_contrast_target

        loss.backward()

        optimizer.step()

        # if scheduler is not None:
        #     scheduler.step()
        #     args.learning_rate = scheduler.get_last_lr()[0]

        if (i_iter % 50 == 0):
            print(exp_dir.split('/')[-1] + '_time = {0},lr = {1: 5f}'.format(datetime.datetime.now(),
                                                   args.learning_rate))
            
            print('iter = {0:8d}/{1:8d}, loss_seg_ce_target = {2:.5f}, loss_seg_dice_target = {3:.5f},'
                 'loss_det_target = {4:.5f}, loss_contrast_target = {5:.5f}'.format(
                    i_iter, args.num_steps, loss_seg_target_value, loss_seg_dice_target_value, 
                    loss_det_target_value, loss_contrast_value))


        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            seg_val_dir = os.path.join(exp_dir  + '/val/R', 'iter_' + str(i_iter), 'seg')
            det_val_dir = os.path.join(exp_dir  + '/val/R', 'iter_' + str(i_iter), 'det')
            make_dirs(seg_val_dir)
            make_dirs(det_val_dir)

            dice, jac = validate_model(model, valloader, seg_val_dir, det_val_dir, args.gpu, usecuda, type='mito')

            print('val dice: %4f' % dice, 'val jac: %4f' % jac)
            if jac > args.best_tjac:
                args.best_tjac = jac
                print('best val dice: %4f' % dice, 'best val jac: %f' % jac)
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir, 'best.pth'))
                
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir, 'best'+ '_' +str(i_iter) + '.pth'))
                           
        scheduler.step(args.best_tjac)
        args.learning_rate = optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    args = get_arguments()

    exp_root_dir = './WSDA_cvlab2R/'
    exp_base_name = 'cvlab2R_15%_stage2_nocontrast_sgd'
    last_exp_dir = './WSDA_cvlab2R/cvlab2R_15%_stage1_nocontrast_sgd/iter4'
    args.restore_from = './WSDA_cvlab2R/cvlab2R_15%_stage1_nocontrast_sgd/iter4/snapshots/best.pth'

    exp_base_dir = exp_root_dir + exp_base_name
    
    # args.learning_rate = 0.00005
    args.gpu = 0

    code_path_list= ['./02_Stage2_WSDA_final.py', './add_arguments.py', './model/HSC82.py',
                     './dataset/data_aug.py', './dataset/source_dataset.py','./dataset/target_dataset.py', 
                     './utils/prototype.py', './val.py', './utils/loss.py']
    bak_code(code_path_list,exp_base_dir)
    
    usecuda = True
    model = CoDA_Net(in_channels=1, out_channels=2, device=args.gpu)
    model.train()

    if usecuda:
        cudnn.benchmark = True
        cudnn.enabled = True
        model.cuda(args.gpu)

    for iter in range(1, 3):

        args.num_steps = 25000
        args.learning_rate = 0.0001
        
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

        point_num = val_threshold(args.restore_from, pseudoseg_save_path, pseudodet_save_path, args, threshold=0.8)
       
        print('============Generate Pseudo Label Start============')

        label_selected_ratio = iter * 0.35 #0.9099
        main(exp_dir, pseudoseg_save_path, pseudodet_save_path, label_selected_ratio)
        last_exp_dir = exp_dir

