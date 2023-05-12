import datetime
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import logging

import cv2

import torch
import torch.utils.data
import torch.optim as optim

import tensorboardX
from tqdm import tqdm

from util import discretized_dataloader
from util.dataset_processing import evaluation
from util import transform, config, transform_graspnet

from skimage.feature import peak_local_max
from model.model_ggcnn.common import post_process_output, post_process_output2

#from model.unet_grasping_divided import UNET_POSITION as fssg_net_pos
#from model.unet_grasping_divided import UNET_ANGLE as fssg_net
from model.backbone_pos_angle_width import BACKBONE, WIDTH_HEAD, ANGLE_HEAD, POS_HEAD
from model.model_asgnet.asgnet_features import AsgnetModel



import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)

config_path = "./config/graspnet/generic.yaml"

def save(args, results, epoch, i, rgb, label, depth, s_x, s_y, pos_hmap, cos_hmap, sin_hmap, widht_hmap):
    f1 = plt.figure(11)
    f1.suptitle("Images")
    f1.add_subplot(2, 2, 1)
    plt.imshow( (rgb.permute(2,3,1,0).detach().cpu().numpy()[:,:,:,0]).astype(np.uint8) )
    f1.add_subplot(2, 2, 2)
    plt.imshow(label.permute(1,2,0).detach().cpu().numpy()[:,:,0])
    f1.add_subplot(2, 2, 3)
    plt.imshow(depth.permute(1,2,3,0).detach().cpu().numpy()[0,:,:,0])
    #f1.savefig(args.save_images_path+str(epoch)+"_"+str(i)+"_"+"epoch_image.png")
    #plt.close()

    f2 = plt.figure(12)
    f2.suptitle("Supports")
    s_x = s_x.permute(1,3,4,2,0).detach().cpu().numpy()[:,:,:,:,0]
    for i,img in enumerate(s_x):
        f2.add_subplot(3, 2, i+1)
        plt.imshow(img.astype(np.uint8))
    #f2.savefig(args.save_images_path+str(epoch)+"_"+str(i)+"_"+"epoch_support.png")
    #plt.close()

    f4 = plt.figure(14)
    f4.suptitle("Supports mask")
    s_y = s_y.permute(1,2,3,0).detach().cpu().numpy()[:,:,:,0]
    for i,img in enumerate(s_y):
        f4.add_subplot(3, 2, i+1)
        plt.imshow(img.astype(np.uint8))
    #f4.savefig(args.save_images_path+str(epoch)+"_"+str(i)+"_"+"epoch_supportmask.png")
    #plt.close()

    f3 = plt.figure(13)
    f3.suptitle("Heatmaps")
    f3.add_subplot(2, 2, 1)
    plt.imshow(pos_hmap.permute(1,2,0).detach().cpu().numpy()[:,:,0])
    f3.add_subplot(2, 2, 2)
    plt.imshow(cos_hmap.permute(1,2,0).detach().cpu().numpy()[:,:,0])
    f3.add_subplot(2, 2, 3)
    plt.imshow(sin_hmap.permute(1,2,0).detach().cpu().numpy()[:,:,0])
    f3.add_subplot(2, 2, 4)
    plt.imshow(widht_hmap.permute(1,2,0).detach().cpu().numpy()[:,:,0])
    #f3.savefig(args.save_images_path+str(epoch)+"_"+str(i)+"_"+"epoch_grasps.png")
    #plt.close()


def get_dataloaders(args):

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    # CREA TRANSFORMS PER IL TRAINING
    train_transform = [
        #transform_graspnet.RandRotateGrasp([args.rotate_min, args.rotate_max], padding=mean,
        #                                   ignore_label=args.padding_label),
        #transform_graspnet.RandScaleGrasp([args.scale_min, args.scale_max]),
        transform_graspnet.RandomGaussianBlurGrasp(),
        transform_graspnet.RandomHorizontalFlipGrasp(),
        transform_graspnet.RandomVerticalFlipGrasp(),
        transform_graspnet.RandomDepthTranslation(),
        transform_graspnet.CropGrasp([721, 721], crop_type='rand', padding=mean, ignore_label=0),
        transform_graspnet.ToTensorGrasp(),
        transform_graspnet.NormalizeGrasp(mean=mean, std=std)
    ]

    train_transform = transform_graspnet.ComposeGrasp(train_transform)
    # CREA TRANSFORMS PER IL SUPPORTO TRAINING
    train_transform_shots = [
        transform.RandScale([args.scale_min_shots, args.scale_max_shots]),
        transform.RandRotate([args.rotate_min_shots, args.rotate_max_shots], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]

    train_transform_shots = transform.Compose(train_transform_shots)

    train_data = discretized_dataloader.GraspingData(shot=args.shot, data_root=args.data_root,
                                                       data_list=args.train_list, transform=train_transform,
                                                       transform_shots=train_transform_shots, mode='train',
                                                       data_classes=args.class_list)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)

    val_transform = transform_graspnet.ComposeGrasp([
        transform_graspnet.RandomHorizontalFlipGrasp(),
        transform_graspnet.RandomVerticalFlipGrasp(),
        transform_graspnet.RandomDepthTranslation(),
        transform_graspnet.CropGrasp([args.train_h, args.train_w], crop_type='custom', padding=mean, ignore_label=args.padding_label),
        transform_graspnet.ToTensorGrasp(),
        transform_graspnet.NormalizeGrasp(mean=mean, std=std)])
    #val_transform_shots = transform.Compose([
    #    transform.Resize(size=args.val_size),
    #    transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
    #    transform.ToTensor(),
    #    transform.Normalize(mean=mean, std=std)])

    val_data = discretized_dataloader.GraspingData(shot=args.shot, data_root=args.data_root,
                                                     data_list=args.val_list, transform=val_transform,
                                                     transform_shots=train_transform_shots, mode='val',
                                                     data_classes=args.class_list)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=val_sampler,
                                             drop_last=True)

    return train_loader, val_loader


def validate(net, device, val_data, batches_per_epoch, args, epoch=0):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct_semantic': 0,
        'failed_semantic': 0,
        'correct_semantic_ARGMAX': 0,
        'failed_semantic_ARGMAX': 0,
        'correct_object': 0,
        'failed_object': 0,
        'loss': 0,
        'losses': {

        },
        'info': {

        }
    }

    ld = len(val_data)

    number = 0
    losses = []

    diff = (1281-721)/2

    with torch.no_grad():
        batch_idx = 0
        #while batch_idx < batches_per_epoch:

        # for x, y, didx, rot, zoom_factor in val_data:
        for i, (rgb, label, depth, s_x, s_y, s_init_seed, subcls_list, pos_hmap, angle_hmap, width_hmap,
            grasp_path, class_chosen)  in enumerate(tqdm(val_data)):

            batch_idx += 1
            if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                break

            depth = depth.float().to(device)
            rgb = rgb.float().to(device)
            label = label.float().to(device)
            s_x = s_x.float().to(device)
            s_y = s_y.float().to(device)
            s_init_seed = s_init_seed.to(device)

            yc_semantic = [pos_hmap, angle_hmap, width_hmap]
            yc_semantic = [yy.long().to(device) for yy in yc_semantic]

            lossd = net.compute_loss(depth, yc_semantic, rgb, label, s_x, s_y, s_init_seed, device, type="val", ep=epoch)

            loss = lossd['loss']

            losses.append(loss.detach().float().cpu())

            # aggiunge dati loss per visualizzazione
            results['loss'] += loss.item() / ld

            # aggiunge dati varie loss
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            for ln, l in lossd['info'].items():
                if ln not in results['info']:
                    results['info'][ln] = 0
                results['info'][ln] += l.item() / ld

            if args.partial_train == 0:
                results['failed_semantic'] += 1
                results['failed_object'] += 1

            
            if args.partial_train == 1:
                no_grasps = 1
                pred = lossd['pred']['pos'][0,:,:]
                #print("PRED: ", pred.shape)
                local_max = peak_local_max(pred.detach().cpu().numpy(), min_distance=20, threshold_abs=0.0, num_peaks=no_grasps)
                #print("Local max: ", local_max)
                if len(local_max) == 0:
                     results['failed_semantic'] += 1
                     results['failed_object'] += 1
                else:
                    val = yc_semantic[0][0, local_max[0][0], local_max[0][1]]
                    if val == 2:
                        results['correct_semantic'] += 1
                        results['correct_object'] += 1
                    else:
                        if val == 1:
                            results['correct_object'] += 1
                        else:
                            results['failed_object'] += 1
                        results['failed_semantic'] += 1


                grasping_point_seq = np.argmax(pred.detach().cpu().numpy())
                grasping_point = [int(grasping_point_seq / 721), int(grasping_point_seq % 721)]
                val = yc_semantic[0][0, grasping_point[0], grasping_point[1]]
                if val == 2:
                    results['correct_semantic_ARGMAX'] += 1
                else:
                    results['failed_semantic_ARGMAX'] += 1
            
            if args.partial_train == 2:
                results['failed_semantic'] += 1
                results['failed_object'] += 1
                results['failed_semantic_ARGMAX'] += 1
            
            if args.partial_train == 3:
                results['failed_semantic'] += 1
                results['failed_object'] += 1
                results['failed_semantic_ARGMAX'] += 1

    
    #visualize losses
    '''
    f5 = plt.figure(0)
    f5.suptitle("Validation Error")
    distribution, bins = np.histogram(losses, bins=20)
    bucket = (bins[1:] - bins[0:-1])/2 + bins[0:-1]
    plt.plot(bucket, distribution)
    path = "/home/bacchin/SemGraspNet/SemGraspNet_venv/FSGGCNN/saved_images/test_{}_{}".format(args.partial_train,"val")
    if not os.path.exists(path):
        print("create follder ",path)
        os.makedirs(path)
    f5.savefig(path + "/"+"epoch_{}_hist.png".format(epoch))
    plt.close(f5)

    losses = np.array(losses)
    '''
    #print("***** VALIDATION LOSS DICT: ", lossd)


    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, args, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'cos_pred': 0,
        'cos_GT': 0,
        'losses': {
        },

        'info': {

        }
    }

    ld = len(train_data)

    print("*** ld train: ",ld)

    number = 0

    net.train()

    batch_idx = 0
    losses = []
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    #while batch_idx < batches_per_epoch:

    # for each data in dataloader
    for i,(rgb, label, depth, s_x, s_y, s_init_seed, subcls_list, pos_hmap, angle_hmap, width_hmap,
            grasp_path, class_chosen) in enumerate(tqdm(train_data)):
        batch_idx += 1
        if batch_idx >= batches_per_epoch:
            break

        optimizer.zero_grad()

        # TRASPORTA TENSORI SU
        depth = depth.float().to(device)
        rgb = rgb.float().to(device)
        label = label.float().to(device)
        s_x = s_x.float().to(device)
        s_y = s_y.float().to(device)
        s_init_seed = s_init_seed.to(device)

        yc_semantic = [pos_hmap, angle_hmap, width_hmap]
        yc_semantic = [yy.long().to(device) for yy in yc_semantic]

        lossd = net.compute_loss(depth, yc_semantic, rgb, label, s_x, s_y, s_init_seed, device, type="train",ep=epoch)

        loss = lossd['loss']
        losses.append(loss.detach().float().cpu())

        # if batch_idx % 100 == 0:
        #    logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, results['loss']))

        results['loss'] += loss.item() / ld
        for ln, l in lossd['losses'].items():
            if ln not in results['losses']:
                results['losses'][ln] = 0
            results['losses'][ln] += l.item() / ld

        for ln, l in lossd['info'].items():
            if ln not in results['info']:
                results['info'][ln] = 0
            results['info'][ln] += l.item() / ld

        loss.backward()
        optimizer.step()

    #visualize losses
    '''
    losses = np.array(losses)
    f5 = plt.figure(0)
    f5.suptitle("Train Error")
    distribution, bins = np.histogram(losses, bins=20)
    bucket = (bins[1:] - bins[0:-1])/2 + bins[0:-1]
    plt.plot(bucket, distribution)
    path = "/home/bacchin/SemGraspNet/SemGraspNet_venv/FSGGCNN/saved_images/test_{}_{}".format(args.partial_train,"train")
    if not os.path.exists(path):
        print("create follder ",path)
        os.makedirs(path)
    f5.savefig(path + "/"+"epoch_{}_hist.png".format(epoch))
    plt.close(f5)
    '''

    #print("***** TRAIN LOSS DICT:", lossd)

    return results


def run():
    args = config.load_cfg_from_cfg_file(config_path)
    print(args)
    # Vis window
    if args.vis:
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, args.description)

    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    train_data, val_data = get_dataloaders(args)
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')

    device = torch.device(args.device)

    if args.partial_train == 0:
        print("TRAINING BACKBONE... ")
        net = BACKBONE(args, vis=args.visualize_model_param)
    else: 
        #LOAD BACKCBONE
        backbone = BACKBONE(args, vis=args.visualize_model_param)
        logging.info('Loading pretrained weights for BACKBONE')
        logging.info("Load from: " + args.weight_backbone)
        checkpoints = torch.load(args.weight_backbone)
        backbone.load_state_dict(checkpoints, strict=True)
        backbone = backbone.to(device)

        if args.partial_train ==1:
            few_shot_model = AsgnetModel(args)
            logging.info('Loading pretrained weights for FEW SHOT MODEL')
            logging.info("Load from: " + args.weight_fs)
            checkpoints = torch.load(args.weight_fs)

            for key in list(checkpoints['state_dict'].keys()):
                #print("OLD KEY: ", key)
                new_key = key[key.find(".") + 1:len(key)]
                #print("NEW KEY: ", new_key)
                checkpoints['state_dict'][new_key] = checkpoints['state_dict'][key]
                del checkpoints['state_dict'][key]

            few_shot_model.load_state_dict(checkpoints['state_dict'], strict=True)
            few_shot_model = few_shot_model.to(device)

            net = POS_HEAD(args, backbone_model=backbone, fs_model=few_shot_model, vis=args.visualize_model_param)

        if args.partial_train ==2:
            net = ANGLE_HEAD(args, backbone, vis=args.visualize_model_param)

        if args.partial_train ==3:
            net = WIDTH_HEAD(args, backbone, vis=args.visualize_model_param)


    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    logging.info('Done')

    
    '''
    if args.weight_model_pos is not None:
        pos_net = fssg_net_pos(args, vis=args.visualize_model_param, loss=args.ang_loss)
        logging.info('Loading pretrained weights')
        logging.info("Load from: " + args.weight_model_pos)
        checkpoints = torch.load(args.weight_model_pos)
        pos_net.load_state_dict(checkpoints, strict=False)

        net = fssg_net(args, pos_net, vis=args.visualize_model_param, loss=args.ang_loss)
    else:
        #print("Using loss: ", args.ang_loss)
        net = fssg_net_pos(args, vis=args.visualize_model_param, loss=args.ang_loss)
    #net = fssg_net(args, vis=args.visualize_model_param, loss=args.ang_loss)
    logging.info(net)
    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    logging.info('Done')


    # Load pretrained 
    if args.weight_model is not None: #the whole model
        logging.info('Loading pretrained weights')
        logging.info("Load from: " + args.weight_model)
        checkpoints = torch.load(args.weight_model)
        net.load_state_dict(checkpoints, strict=False)
    else: #only pretrained fs and backbone
        logging.info('Loading pretrained weights')
        logging.info("Load fs from: " + args.weight_fs)

        checkpoints_fs = torch.load(args.weight_fs)
        # net.load_state_dict(checkpoints_fs['state_dict'])

        for key in list(checkpoints_fs['state_dict'].keys()):
            new_key = "few_shot_model." + key[key.find(".") + 1:len(key)]
            checkpoints_fs['state_dict'][new_key] = checkpoints_fs['state_dict'][key]
            del checkpoints_fs['state_dict'][key]
        net.load_state_dict(checkpoints_fs['state_dict'], strict=False)


        if args.weight_backbone is not None:
            logging.info("Load gg from: " + args.weight_backbone)
            checkpoints_backbone = torch.load(args.weight_backbone)
            for key in list(checkpoints_backbone.keys()):
                if key.split('.')[0] != 'resnet50_backbone' and key.split('.')[0] != 'decoder_feature':
                    del checkpoints_backbone[key]
            net.load_state_dict(checkpoints_backbone, strict=False)

        logging.info("Done")

        '''


    best_iou = 0.0

    if args.save_crit == 'val_error':
        th = float('inf')
    if args.save_crit == 'iou':
        th = 0.0
    save_flag = False

    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, args, vis=args.vis)

        logging.info('Epoch: {}, Loss: {:0.4f}'.format(epoch, train_results['loss']))

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)

        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        for n, l in train_results['info'].items():
            tb.add_scalar('train_info/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.batches_per_epoch, args,epoch=epoch)
        logging.info('Correct grasps: %d/%d = %f' % (test_results['correct_semantic'], test_results['correct_semantic'] + test_results['failed_semantic'],
                                     test_results['correct_semantic'] / (test_results['correct_semantic'] + test_results['failed_semantic'])))

        # Log validation results to tensorbaord
        # Log valid grasps
        tb.add_scalar('CORRECT/val_semantic', test_results['correct_semantic'] / (test_results['correct_semantic'] +
                                                                              test_results['failed_semantic']), epoch)
        tb.add_scalar('CORRECT/val_objects', test_results['correct_object'] / (test_results['correct_object'] +
                                                                               test_results['failed_object']), epoch)
        
        tb.add_scalar('CORRECT/val_semantic_ARGMAX', test_results['correct_semantic_ARGMAX'] /
                      (test_results['correct_semantic_ARGMAX'] + test_results['failed_semantic_ARGMAX']), epoch)

        '''
        tb.add_scalar('CORRECT/val_semantic_to', test_results['correct_semantic_to'] /
                      (test_results['correct_semantic_to'] + test_results['failed_semantic_to']), epoch)
        tb.add_scalar('CORRECT/val_semantic_tob', test_results['correct_semantic_tob'] /
                      (test_results['correct_semantic_tob'] + test_results['failed_semantic_tob']), epoch)
        '''
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        for n, l in test_results['info'].items():
            tb.add_scalar('val_info/' + n, l, epoch)

        # Save best performing network
        if args.save_crit == 'val_error':
            val = test_results['loss']
            if val < th:
                th = val
                save_flag = True

        if args.save_crit == 'iou':
            val = test_results['correct_semantic'] / (test_results['correct_semantic'] + test_results['failed_semantic'])
            if val > th:
                th = val
                save_flag = True

        iou =  test_results['correct_semantic_ARGMAX'] / (test_results['correct_semantic_ARGMAX'] + test_results['failed_semantic_ARGMAX'])
        if save_flag or epoch == 0 or (epoch % args.save_every) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_%s_%0.2f' % (epoch, args.save_crit , val)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_%s_%0.2f_statedict.pt' % (epoch, args.save_crit, val)))
            save_flag = False


if __name__ == '__main__':
    run()
