import torch.nn as nn
import torch.nn.functional as F
import torch
import os

from model.model_asgnet.asgnet_features import AsgnetModel
from model.model_unet.unet_grasping import DoubleConv, Down, Up, OutConv
from skimage.feature import peak_local_max
import numpy as np
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class WIDTH_HEAD(nn.Module):

    def __init__(self, args, backbone_model, input_channels=3, vis=False, loss=0):
        super(WIDTH_HEAD, self).__init__()

        self.backbone_model = backbone_model
        if backbone_model is not None:
            self.backbone_model.eval()
        self.partial_train = args.partial_train

        self.n_classes = 16
        self.input_size = 224

        bilinear = True
        #filters_size  = [input_channels, 32, 64, 128, 256, 512, self.n_classes] #used for th last training
        filters_size  = [input_channels, 16, 32, 64, 128, 256, self.n_classes] 

        self.inc = DoubleConv(filters_size[0], filters_size[1])
        self.down1 = Down(filters_size[1], filters_size[2])
        self.down2 = Down(filters_size[2], filters_size[3])
        self.down3 = Down(filters_size[3], filters_size[4])
        factor = 2 if bilinear else 1
        self.down4 = Down(filters_size[4], filters_size[5] // factor)
        self.up1 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4 = Up(filters_size[2], filters_size[1], bilinear)
        self.outc = OutConv(filters_size[1], filters_size[-1])

        self.counter = 0

    def forward(self, x):
        self.counter += 1

        i0 = F.interpolate(x[:, 0, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
        i1 = F.interpolate(x[:, 1, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
        i2 = F.interpolate(x[:, 2, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)

        fs_out = torch.cat((i0, i1, i2), 1).cuda()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        new_out = []
        for k in range(self.n_classes):
            out = F.interpolate(logits[:, k, :, :].unsqueeze(1), size=(721, 721), mode='bilinear', align_corners=False)
            new_out.append(out)
        out = torch.cat(new_out, 1).cuda()

        return out

    def compute_loss(self, xc, yc_grasp, rgb_x, rgb_y, support_x, support_y, s_seed, dev, type="train", ep=0):

        y_pos, y_angle_grasp, y_width_grasp = yc_grasp

        # GET POSITION
        with torch.no_grad():
            features = self.backbone_model(xc)

        # GET POSITION AND CROPPED IMAGE
        new_input_tensor = []
        for k in range(y_pos.shape[0]):

            new_input_tensor.append(torch.cat((xc[k,:,:,:].unsqueeze(0), features[k,0,:,:].unsqueeze(0).unsqueeze(0), features[k,1,:,:].unsqueeze(0).unsqueeze(0)), dim=1))

        new_input_tensor = torch.cat(new_input_tensor, dim=0)

        # ANGEL INFERENCE
        width_pred = self(new_input_tensor)

        w = torch.tensor([0.1,2.0,2.0,2.0,2.0,2.0,2.0,1.5,1.0,1.0,1.0,1.0,0.8,0.8,0.8,0.3]).float().cuda() #DA DOVE VIENE FUORI? 
        
        width_loss_grasp = F.cross_entropy(width_pred.float(), y_width_grasp.long(), weight=w)

        loss = width_loss_grasp

        if type=="train":
            save_list = range(1,913,90)
            norm = 913
        if type=="val":
            save_list = range(1,1506,150)
            norm = 1506

        if self.counter >= (18 * 0) and self.counter%norm in save_list:
        
            for b in range(y_width_grasp.shape[0]):
                f5 = plt.figure(0, figsize=(10,10))
                f5.suptitle("Images")
                f5.add_subplot(3, 3, 1)
                plt.imshow((y_width_grasp[b, :, :].detach().cpu().numpy()))
                f5.add_subplot(3, 3, 2)
                plt.imshow((xc[b, 0, :, :].detach().cpu().numpy()))
                #f5.add_subplot(3, 3, 3)
                #plt.imshow(torch.argmax(y_pos.clone(), dim=1).detach().cpu().numpy()[0, :, :])
                ax0 = f5.add_subplot(3, 3, 4)
                im = plt.imshow(torch.argmax(width_pred.clone(), dim=1).detach().cpu().numpy()[b, :, :])
                ax1 = f5.add_subplot(3, 3, 5)
                f5.colorbar(im,ax=ax0)
                im1 = plt.imshow((width_pred[b, 2, :, :].clone().detach().cpu().numpy()))
                ax1.set_title("LAYER BK - 0")
                f5.colorbar(im1, ax=ax1)
                ax2 = f5.add_subplot(3, 3, 6)
                im2 = plt.imshow((width_pred[b, 6, :, :].clone().detach().cpu().numpy()))
                ax2.set_title("LAYER OBJ - 1")
                f5.colorbar(im2, ax=ax2)
                ax3 = f5.add_subplot(3, 3, 7)
                im3 = plt.imshow((width_pred[b, 12, :, :].clone().detach().cpu().numpy()))
                ax3.set_title("LAYER TARGET - 2")
                f5.colorbar(im3, ax=ax3)
                ax4 = f5.add_subplot(3, 3, 8)
                im4 = plt.imshow((width_pred[b, 13, :, :].clone().detach().cpu().numpy()))
                f5.colorbar(im4, ax=ax4)
                ax5 =f5.add_subplot(3, 3, 9)
                im5 = plt.imshow((width_pred[b, 15, :, :].clone().detach().cpu().numpy()))
                f5.colorbar(im5, ax=ax5)
                path = "/home/bacchin/SemGraspNet/SemGraspNet_venv/FSGGCNN/saved_images/test_{}_{}".format(self.partial_train,type)
                if not os.path.exists(path):
                    print("create follder ",path)
                    os.makedirs(path)
                f5.savefig(path + "/"+"epoch_{}_iter_{}_image.png".format(ep,self.counter%norm))
                plt.close(f5)
                #plt.show()
        return {
            'loss': loss,
            'losses': {
                # 'p_loss': p_loss,
                # 'cos_loss': cos_loss,
                # 'sin_loss': sin_loss,
                # 'width_loss': width_loss
            },
            'info': {
            },
            'pred': {
                'pos': y_pos,
                # 'cos': cos_pred,
                # 'sin': sin_pred,
                'angle': y_angle_grasp,
                'width': width_pred
            },
            'pretrained': {
                # 'few_shot': fs_out,
                # 'grasp': res_before
            }
        }


class ANGLE_HEAD(nn.Module):

    def __init__(self, args, backbone_model, input_channels=3, vis=False, loss=0):
        super(ANGLE_HEAD, self).__init__()

        self.backbone_model = backbone_model
        if backbone_model is not None:
            self.backbone_model.eval()
        self.partial_train = args.partial_train

        self.n_classes = 18
        self.input_size = 224

        bilinear = True
        #filters_size  = [input_channels, 4, 8, 16, 32, 64, self.n_classes]
        filters_size  = [input_channels, 32, 64, 128, 256, 512, self.n_classes] #used for last training
        #filters_size  = [input_channels, 16, 32, 64, 128, 256, self.n_classes] 

        self.inc = DoubleConv(filters_size[0], filters_size[1])
        self.down1 = Down(filters_size[1], filters_size[2])
        self.down2 = Down(filters_size[2], filters_size[3])
        self.down3 = Down(filters_size[3], filters_size[4])
        factor = 2 if bilinear else 1
        self.down4 = Down(filters_size[4], filters_size[5] // factor)
        self.up1 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4 = Up(filters_size[2], filters_size[1], bilinear)
        self.outc = OutConv(filters_size[1], filters_size[-1])

        self.counter = 0

    def forward(self, x):
        self.counter += 1

        i0 = F.interpolate(x[:, 0, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
        i1 = F.interpolate(x[:, 1, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
        i2 = F.interpolate(x[:, 2, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)

        fs_out = torch.cat((i0, i1, i2), 1).cuda()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        new_out = []
        for k in range(self.n_classes):
            out = F.interpolate(logits[:, k, :, :].unsqueeze(1), size=(721, 721), mode='bilinear', align_corners=False)
            new_out.append(out)
        out = torch.cat(new_out, 1).cuda()

        return out

    def compute_loss(self, xc, yc_grasp, rgb_x, rgb_y, support_x, support_y, s_seed, dev, type="train", ep=0):

        y_pos, y_angle_grasp, y_width_grasp = yc_grasp

        # GET POSITION
        with torch.no_grad():
            features = self.backbone_model(xc)

        # NEW INPUT TENSOR
        new_input_tensor = []
        for k in range(y_pos.shape[0]):
            new_input_tensor.append(torch.cat((xc[k,:,:,:].unsqueeze(0), features[k,0,:,:].unsqueeze(0).unsqueeze(0), features[k,1,:,:].unsqueeze(0).unsqueeze(0)), dim=1))

        new_input_tensor = torch.cat(new_input_tensor, dim=0)

        # ANGEL INFERENCE
        angle_pred = self(new_input_tensor)

        w = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.2,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]).float().cuda()
        angle_loss_grasp = F.cross_entropy(angle_pred.float(), y_angle_grasp.long(), weight=w)

        loss = angle_loss_grasp

        if type=="train":
            save_list = range(1,913,90)
            norm = 913
        if type=="val":
            save_list = range(1,1506,150)
            norm = 1506

        if self.counter >= (18 * 0) and self.counter%norm in save_list:
            f5 = plt.figure(0, figsize=(10,10))
            f5.suptitle("Images")
            f5.add_subplot(3, 3, 1)
            plt.imshow((y_angle_grasp[0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 3, 2)
            plt.imshow((xc[0, 0, :, :].detach().cpu().numpy()))
            #f5.add_subplot(3, 3, 3)
            #plt.imshow(torch.argmax(y_pos.clone(), dim=1).detach().cpu().numpy()[0, :, :])
            f5.add_subplot(3, 3, 4)
            plt.imshow(torch.argmax(angle_pred.clone(), dim=1).detach().cpu().numpy()[0, :, :])
            ax1 = f5.add_subplot(3, 3, 5)
            im1 = plt.imshow((angle_pred[0, 4, :, :].clone().detach().cpu().numpy()))
            ax1.set_title("LAYER BK - 0")
            f5.colorbar(im1, ax=ax1)
            ax2 = f5.add_subplot(3, 3, 6)
            im2 = plt.imshow((angle_pred[0, 8, :, :].clone().detach().cpu().numpy()))
            ax2.set_title("LAYER OBJ - 1")
            f5.colorbar(im2, ax=ax2)
            ax3 = f5.add_subplot(3, 3, 7)
            im3 = plt.imshow((angle_pred[0, 9, :, :].clone().detach().cpu().numpy()))
            ax3.set_title("LAYER TARGET - 2")
            f5.colorbar(im3, ax=ax3)
            ax4 = f5.add_subplot(3, 3, 8)
            im4 = plt.imshow((angle_pred[0, 10, :, :].clone().detach().cpu().numpy()))
            f5.colorbar(im4, ax=ax4)
            ax5 =f5.add_subplot(3, 3, 9)
            im5 = plt.imshow((angle_pred[0, 14, :, :].clone().detach().cpu().numpy()))
            f5.colorbar(im5, ax=ax5)
            path = "/home/bacchin/SemGraspNet/SemGraspNet_venv/FSGGCNN/saved_images/test_{}_{}".format(self.partial_train,type)
            if not os.path.exists(path):
                print("create follder ",path)
                os.makedirs(path)
            f5.savefig(path + "/"+"epoch_{}_iter_{}_image.png".format(ep,self.counter%norm))
            plt.close(f5)

        return {
            'loss': loss,
            'losses': {
                # 'p_loss': p_loss,
                # 'cos_loss': cos_loss,
                # 'sin_loss': sin_loss,
                # 'width_loss': width_loss
            },
            'info': {
            },
            'pred': {
                'pos': y_pos,
                # 'cos': cos_pred,
                # 'sin': sin_pred,
                'angle': angle_pred,
                'width': y_width_grasp
            },
            'pretrained': {
                # 'few_shot': fs_out,
                # 'grasp': res_before
            }
        }


class POS_HEAD(nn.Module):

    def __init__(self, args, backbone_model, fs_model, input_channels=4, vis=False, loss=0):
        super(POS_HEAD, self).__init__()

        self.counter = 0

        self.few_shot_model = fs_model 
        self.few_shot_model.eval()
        self.partial_train = args.partial_train

        # FREEZE ASGNET
        for param in self.few_shot_model.parameters():
            param.requires_grad = False


        self.backbone_model = backbone_model

        self.n_classes = 1
        bilinear = True
        factor = 2 if bilinear else 1

        #filters_size_down  = [input_channels, 8, 16, 24, 32, 40] #2
        filters_size_down  = [input_channels, 8, 10, 12, 16, 20]  #1
        filters_size_up  = [ filters_size_down[5] // factor+filters_size_down[4], 
                            filters_size_down[4] // factor+filters_size_down[3],
                            filters_size_down[3] // factor+filters_size_down[2],
                            filters_size_down[2] // factor+filters_size_down[1],
                            self.n_classes]

        self.inc = DoubleConv(filters_size_down[0], filters_size_down[1])
        #self.down1 = Down(filters_size_down[1], filters_size_down[2])
        #self.down2 = Down(filters_size_down[2], filters_size_down[3])
        self.down3 = Down(filters_size_down[1], filters_size_down[2])
        
        self.down4 = Down(filters_size_down[2], filters_size_down[3] // factor)
        #self.up1 = Up(filters_size_up[1], filters_size_down[3] // factor, bilinear)
        #self.up2 = Up(filters_size_up[2], filters_size_down[2] // factor, bilinear)
        self.up3 = Up(filters_size_up[-3], filters_size_down[2] // factor, bilinear)
        self.up4 = Up(filters_size_up[-2], filters_size_down[1], bilinear)
        self.outc = OutConv(filters_size_down[1], filters_size_up[-1])

        '''
        self.inc = DoubleConv(input_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        #self.down3 = Down(16, 24)
        factor = 2 if bilinear else 1
        self.down4 = Down(32, 64 // factor)
        self.up1 = Up(64, 32 // factor, bilinear)
        self.up2 = Up(32, 16 // factor, bilinear)
        #self.up3 = Up(16, 8 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)
        '''

        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.2)
        self.drop4 = nn.Dropout(p=0.2)

    def forward(self, features, rgb_x, support_x, support_y, s_seed):
        
        self.counter += 1
        _, fs_out = self.few_shot_model(rgb_x, s_x=support_x, s_y=support_y, s_seed=s_seed)

        fs = fs_out.clone()

        fs_out_1 = F.interpolate(fs_out[:,1,:,:].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
        fs_out_2 = F.interpolate(fs_out[:, 0, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
        feat_1 = F.interpolate(features[:,1,:,:].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
        feat_2 = F.interpolate(features[:, 0, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)

        fs_out = torch.cat((feat_1.detach(), feat_2.detach(), fs_out_1.detach(), fs_out_2.detach()), 1).cuda()

        x1 = self.inc(fs_out)
        #x2 = self.down1(x1)
        #x2 = self.drop1(x2)
        #x3 = self.down2(x2)
        x3 = self.down3(x1)
        x4 = self.down4(x3)
        #x5 = self.drop2(x5)
        #x = self.up1(x5, x3)
        #x = self.drop3(x)
        #x = self.up2(x, x2)
        x = self.up3(x4, x3)
        x = self.up4(x, x1)
        #x = self.drop4(x)
        logits = self.outc(x)
        out = F.interpolate(logits[:, 0, :, :].unsqueeze(1), size=(721, 721), mode='bilinear', align_corners=False)

        return out, fs

    def compute_loss(self, xc, yc_grasp, rgb_x, rgb_y, support_x, support_y, s_seed, dev, type="train", ep=0):

        y_pos, y_angle_grasp, y_width_grasp = yc_grasp

        #y_pos[y_pos==1] = 0
        #y_pos[y_pos==2] = 1

        # GET POSITION
        if backbone_model is not None:
            with torch.no_grad():
                features = self.backbone_model(xc)

        pos_pred, fs_out = self(features, rgb_x, support_x, support_y, s_seed)

        pos_mask = y_pos.clone().float()

        pos_mask[pos_mask==1] = 0.25
        pos_mask[pos_mask==2] = 1
        foreground = pos_mask == 1
        
        blurrer = T.GaussianBlur(kernel_size=(55, 55), sigma=(18, 18))
        pos_mask = blurrer(pos_mask)
        #pos_mask = blurrer(pos_mask)
        
        pos_pred = pos_pred.squeeze(1)
        

        pos_loss =  F.smooth_l1_loss(pos_pred.float(), pos_mask.float())

        
        size2 = pos_mask[foreground].flatten().shape[0]
        #size3 = pos_mask[~foreground].flatten().shape[0]
        #size = pos_mask.flatten().shape[0]

        w_fg = 0.6
        w_bg = 1 - w_fg

        if size2 == 0.0:
            mse_bg = F.smooth_l1_loss(pos_pred.float(), pos_mask.float())
            mse_fg = torch.tensor(0.0)
            #pos_loss = w_bg * mse_bg
        else:
            #ratio = size3/size2
            mse_fg = F.smooth_l1_loss(pos_pred[foreground].float(), pos_mask[foreground].float())
            mse_bg = F.smooth_l1_loss(pos_pred[~foreground].float(), pos_mask[~foreground].float())
            #pos_loss = w_fg * mse_fg +  w_bg * mse_bg

        #print("pos mask:", pos_mask.shape)
        #print("pos pred:", pos_pred.shape)
        #print("foreground:", foreground.shape)

        loss = pos_loss

        if type=="train":
            norm = 732
            save_list = range(1,norm,70)
        if type=="val":
            norm = 1167
            save_list = range(1,norm,110)

        pred = pos_pred[0,:,:]
        local_max = peak_local_max(pred.detach().cpu().numpy(), min_distance=20, threshold_abs=0.0, num_peaks=1)
        grasping_point_seq = np.argmax(pred.detach().cpu().numpy())
        grasping_point = [int(grasping_point_seq / 721), int(grasping_point_seq % 721)]

        if self.counter >= (18 * 0) and self.counter%norm in save_list:
            f5 = plt.figure(0, figsize=(10,10))
            f5.suptitle("Images")
            ax = f5.add_subplot(3, 3, 1)
            plt.imshow((pos_mask[0, :, :].detach().cpu().numpy()))
            plt.plot(local_max[0][1], local_max[0][0], '+k')
            plt.plot(grasping_point[1], grasping_point[0], 'xw')
            ax.set_title("POS GT")
            f5.add_subplot(3, 3, 2)
            plt.imshow((xc[0, 0, :, :].detach().cpu().numpy()))
            ax = f5.add_subplot(3, 3, 3)
            plt.imshow(torch.argmax(fs_out.clone(), dim=1).detach().cpu().numpy()[0, :, :])
            ax.set_title("FEW SHOT SEG")
            ax = f5.add_subplot(3, 3, 4)
            plt.imshow(torch.argmax(features.clone(), dim=1).detach().cpu().numpy()[0, :, :])
            ax.set_title("BACKBONE SEG")
            ax1 =f5.add_subplot(3, 3, 5)
            im1 = plt.imshow((pos_pred[0, :, :].clone().detach().cpu().numpy()))
            ax1.set_title("POS PRED")
            plt.plot(local_max[0][1], local_max[0][0], '+k')
            plt.plot(grasping_point[1], grasping_point[0], 'xw')
            f5.colorbar(im1, ax=ax1)
            ax1 =f5.add_subplot(3, 3, 6)
            im1 = plt.imshow((fs_out[0, 0, :, :].clone().detach().cpu().numpy()))
            ax1.set_title("FS HMAP 0")
            f5.colorbar(im1, ax=ax1)
            ax1 =f5.add_subplot(3, 3, 7)
            im1 = plt.imshow((fs_out[0, 1, :, :].clone().detach().cpu().numpy()))
            ax1.set_title("FS HMAP 1")
            f5.colorbar(im1, ax=ax1)
            ax1 =f5.add_subplot(3, 3, 8)
            ax1.axis([0, 10, 0, 10])
            ax1.text(2, 9, 'Local Max: pred @[{}, {}] = {}'.format(local_max[0][0], local_max[0][1], pred[local_max[0][0],local_max[0][1]]), style='italic')
            ax1.text(2, 5, 'argmax: pred @[{}, {}] = {}'.format(grasping_point[0], grasping_point[1], pred[grasping_point[0], grasping_point[1]], style='italic'))
            #ax1.text(2, 3, 'span pred: min= {}, max= {}'.format(torch.min(pred),torch.max(pred)), style='italic')

            path = "/home/bacchin/SemGraspNet/SemGraspNet_venv/FSGGCNN/saved_images/test_{}_{}".format(self.partial_train,type)
            if not os.path.exists(path):
                print("create follder ",path)
                os.makedirs(path)
            f5.savefig(path + "/"+"epoch_{}_iter_{}_image.png".format(ep,self.counter%norm))
            plt.close(f5)

        return {
            'loss': loss,
            'losses': {
                #'p_loss': p_loss,
                #'cos_loss': cos_loss,
                #'sin_loss': sin_loss,
                #'width_loss': width_loss
            },
            'info': {
                #'size2': torch.tensor([size2]),
                #'size3': torch.tensor([size3]),
                'err_bg': mse_bg,
                'err_fg': mse_fg
            },
            'pred': {
                'pos': pos_pred,
                #'cos': cos_pred,
                #'sin': sin_pred,
                'angle': y_angle_grasp,
                'width': y_width_grasp
            },
            'pretrained':{
                #'few_shot': fs_out,
                #'grasp': res_before
            }
        }




class BACKBONE(nn.Module):

    def __init__(self, args, input_channels=1, out_classes=2, vis=False, loss=0):
        super(BACKBONE, self).__init__()

        self.counter = 0
        self.partial_train = args.partial_train

        bilinear = True

        filters_size  = [input_channels, 8, 16, 32, 64, 128, out_classes]
        #filters_size  = [input_channels, 16, 32, 64, 128, 256, out_classes] 

        self.inc = DoubleConv(filters_size[0], filters_size[1])
        self.down1 = Down(filters_size[1], filters_size[2])
        self.down2 = Down(filters_size[2], filters_size[3])
        self.down3 = Down(filters_size[3], filters_size[4])
        factor = 2 if bilinear else 1
        self.down4 = Down(filters_size[4], filters_size[5] // factor)
        self.up1 = Up(filters_size[5], filters_size[4] // factor, bilinear)
        self.up2 = Up(filters_size[4], filters_size[3] // factor, bilinear)
        self.up3 = Up(filters_size[3], filters_size[2] // factor, bilinear)
        self.up4 = Up(filters_size[2], filters_size[1], bilinear)
        self.outc = OutConv(filters_size[1], filters_size[-1])

    def forward(self, depth_x):
        self.counter += 1
    
        depth_x = F.interpolate(depth_x[:, 0, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)

        fs_out = depth_x

        x1 = self.inc(fs_out)
        x2 = self.down1(x1)
        #x2 = self.drop1(x2)
        x3 = self.down2(x2)
        #x3 = self.drop2(x3)
        x4 = self.down3(x3)
        #x4 = self.drop3(x4)
        x5 = self.down4(x4)
        #x5 = self.drop4(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        out_0 = F.interpolate(logits[:, 0, :, :].unsqueeze(1), size=(721, 721), mode='bilinear', align_corners=False)
        out_1 = F.interpolate(logits[:, 1, :, :].unsqueeze(1), size=(721, 721), mode='bilinear', align_corners=False)
        out = torch.cat((out_0, out_1), 1).cuda()

        return out

    def compute_loss(self, xc, yc_grasp, rgb_x, rgb_y, support_x, support_y, s_seed, dev, type="train", ep=0):

        y_pos, y_angle_grasp, y_width_grasp = yc_grasp

        y_pos[y_pos==2] = 1

        pred = self(xc)

        obj_seg_mask = y_pos

        pos_loss = F.cross_entropy(pred, obj_seg_mask.long())

        loss = pos_loss

        if type=="train":
            save_list = range(1,456,50)
            norm = 456
        if type=="val":
            save_list = range(1,1506,150)
            norm = 1506


        if self.counter >= (18 * 0) and self.counter%norm in save_list:
            f5 = plt.figure(0, figsize=(10,10))
            f5.suptitle("Images")
            f5.add_subplot(3, 3, 1)
            plt.imshow((y_pos[0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 3, 2)
            plt.imshow((xc[0, 0, :, :].detach().cpu().numpy()))
            ax3 = f5.add_subplot(3, 3, 3)
            im3 = plt.imshow(torch.argmax(pred.clone(), dim=1).detach().cpu().numpy()[0, :, :])
            f5.colorbar(im3, ax=ax3)
            f5.add_subplot(3, 3, 4)
            plt.imshow(obj_seg_mask[0, :, :].detach().cpu().numpy())
            ax1 =f5.add_subplot(3, 3, 5)
            im1 = plt.imshow((pred[0, 0, :, :].clone().detach().cpu().numpy()))
            ax1.set_title("LAYER BG - 0")
            f5.colorbar(im1, ax=ax1)
            ax2 = f5.add_subplot(3, 3, 6)
            im2 = plt.imshow((pred[0, 1, :, :].clone().detach().cpu().numpy()))
            ax2.set_title("LAYER OBJ - 1")
            f5.colorbar(im2, ax=ax2)
            path = "/home/bacchin/SemGraspNet/SemGraspNet_venv/FSGGCNN/saved_images/test_{}_{}".format(self.partial_train,type)
            if not os.path.exists(path):
                print("create follder ",path)
                os.makedirs(path)
            f5.savefig(path + "/"+"epoch_{}_iter_{}_image.png".format(ep,self.counter%norm))
            plt.close(f5)

        return {
            'loss': loss,
            'losses': {
                #'p_loss': p_loss,
                #'cos_loss': cos_loss,
                #'sin_loss': sin_loss,
                #'width_loss': width_loss
            },
            'info': {
            },
            'pred': {
                'obj_feat': pred
            },
            'pretrained':{
                #'few_shot': fs_out,
                #'grasp': res_before
            }
        }



