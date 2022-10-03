import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from torchvision.transforms.functional import crop

from model.model_asgnet.asgnet_features import AsgnetModel 
from model.model_ggcnn.ggcnn_features import GGCNN
from model.external_modules import normalize_fs_out, MLP

from skimage.filters import gaussian
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt

import numpy as np
from model.efficientnet_v2 import EfficientNetV2

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class FusionLayer(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        filter_sizes = [16, 8, 8, 16]
        kernel_sizes = [5, 3, 3, 5]
        strides = [2, 2, 2, 2]

        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=3, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(filter_sizes[3], 1, kernel_size=2)
        )
        for m in self.fusion:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)


    def forward(self, x):
        return self.fusion(x)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
class PROVA_ANGOLO(nn.Module):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """

    def __init__(self, args, position_model, input_channels=3, vis=False, loss=0):
        super(PROVA_ANGOLO, self).__init__()

        self.position_model = position_model

        n_classes = 18
        self.input_size = 224
        """
        self.classifcation_model = models.resnet101(pretrained=False)
        set_parameter_requires_grad(self.classifcation_model, False)
        num_ftrs = self.classifcation_model.fc.in_features
        self.classifcation_model.fc = nn.Linear(num_ftrs, n_classes)
        self.input_size = 224
        """
        self.classifcation_model = EfficientNetV2('m', in_channels=3, n_classes=n_classes, pretrained=False)

        self.counter = 0

    def forward(self, cropped):
        self.counter += 1
        out = self.classifcation_model(cropped)
        return out

    def compute_loss(self, xc, yc, yc_grasp, rgb_x, rgb_y, support_x, support_y, s_seed, dev):
        # SET TENSORS
        y_pos, y_cos, y_sin, y_width = yc
        y_pos = y_pos.float()
        y_cos = y_cos.float()
        y_sin = y_sin.float()
        y_width = y_width.float()

        y_pos_grasp, y_cos_grasp, y_sin_grasp, y_width_grasp = yc_grasp
        y_pos_grasp = y_pos_grasp.float()
        y_cos_grasp = y_cos_grasp.float()
        y_sin_grasp = y_sin_grasp.float()
        y_width_grasp = y_width_grasp.float()

        xc = xc.float()
        rgb_x = rgb_x.float()
        support_x = support_x.float()
        support_y = support_y.float()

        # GET POSITION
        with torch.no_grad():
            pos_pred, fs1, fs2 = self.position_model(xc, rgb_x, support_x, support_y, s_seed)

        #CREATE FINAL HEATMAP
        final_heatmap = pos_pred[:,2,:,:].clone()
        final_heatmap += pos_pred[:,1,:,:].clone()
        final_heatmap += fs1[:,0,:,:].clone()
        final_heatmap -= pos_pred[:,0,:,:].clone()
        final_heatmap -= fs2[:,0,:,:].clone()

        #GET POSITION AND CROPPED IMAGE
        max_indexes = []
        new_input_tensor = []
        crop_dim = int(self.input_size/2)
        for k in range(y_pos.shape[0]):
            img = final_heatmap[k, :, :]
            local_max = torch.argmax(img.clone())
            local_max = [local_max.cpu()/y_pos.shape[1], local_max.cpu()%y_pos.shape[1]]
            top = max(0, local_max[0]-crop_dim)
            left = max(0,local_max[1]-crop_dim)
            top = min(top, y_pos.shape[1]-self.input_size-1)
            left = min(left, y_pos.shape[1]-self.input_size-1)
            top = int(top)
            left = int(left)
            #new_input_tensor.append(crop(torch.cat((xc[k,:,:,:].unsqueeze(0), fs1[k,:,:,:].unsqueeze(0), fs2[k,:,:,:].unsqueeze(0)), dim=1), top, left, self.input_size, self.input_size))
            new_input_tensor.append(crop(torch.cat((xc[k,:,:,:].unsqueeze(0), pos_pred[k,1,:,:].unsqueeze(0).unsqueeze(0), pos_pred[k,2,:,:].unsqueeze(0).unsqueeze(0)), dim=1), top, left, self.input_size, self.input_size))
            max_indexes.append(local_max)

        max_indexes = np.array(max_indexes)  # per ogni batch mi trovo il punto di max della heatmap pos
        new_input_tensor = torch.cat(new_input_tensor, dim=0)

        # ANGEL INFERENCE
        angle_pred = self(new_input_tensor)

        #COMPUTE ANGLE GT
        # Denormalize
        y_sin_grasp = y_sin_grasp * 2 + (-1)
        y_cos_grasp = y_cos_grasp * 2 + (-1)

        cos_GT = torch.empty(y_cos_grasp.shape[0], device=dev)
        sin_GT = torch.empty(y_sin_grasp.shape[0], device=dev)
        w_GT = torch.empty(y_width_grasp.shape[0], device=dev)

        for k in range(y_cos_grasp.shape[0]):
            cos_GT[k] = y_cos_grasp[k, int(max_indexes[k, 0]), int(max_indexes[k, 1])]
            sin_GT[k] = y_sin_grasp[k, int(max_indexes[k, 0]), int(max_indexes[k, 1])]
            w_GT[k] = y_width_grasp[k, int(max_indexes[k, 0]), int(max_indexes[k, 1])]

        # angle creation and normalization
        angle_gt = (torch.atan2(sin_GT, cos_GT) / 2.0)
        angle_gt = (angle_gt + math.pi / 2) / math.pi
        angle_gt_vec = torch.empty((y_cos_grasp.shape[0], 18), device=dev).float()
        angle_gt_vec[:, :] = 0
        index_gt = angle_gt * 18 - 1 / 36
        index_gt = index_gt.long()

        for k in range(y_cos_grasp.shape[0]):
            angle_gt_vec[k, index_gt[k]] = 1

        # pos_pred = y_pos
        # width_pred = y_width

        # print("GT SHAPE", angle_gt.shape)
        # angle_loss_grasp = F.mse_loss(angle_gt, angle[:])
        # angle_loss_grasp = F.cross_entropy(angle.float(), angle_gt_vec.long())
        angle_loss_grasp = F.cross_entropy(angle_pred,index_gt)
        # angle_loss_grasp = F.mse_loss(angle, angle_gt_vec)
        # pos_loss = F.mse_loss(pos_pred, y_pos, reduction='sum')
        #pos_mask = y_pos.clone()
        #pos_mask[pos_mask == 1] = 2
        #pos_mask[pos_mask == 0.25] = 1
        # print(pos_pred.shape)
        # print(pos_mask.shape)
        #pos_loss = F.cross_entropy(pos_pred, pos_mask.long())

        res_angle = torch.argmax(angle_pred, axis=1) / 18 + 1 / 36

        loss = angle_loss_grasp

        if self.counter >= (18 * 0) and self.counter % 180 == 1 and False:
            f5 = plt.figure(5)
            f5.suptitle("Images")
            f5.add_subplot(3, 3, 1)
            plt.imshow((y_pos[0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 3, 2)
            plt.imshow((xc[0, 0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 3, 3)
            plt.imshow(torch.argmax(pos_pred.clone(), dim=1).detach().cpu().numpy()[0, :, :])
            f5.add_subplot(3, 3, 4)
            plt.imshow(pos_mask[0, :, :].detach().cpu().numpy())
            ax1 = f5.add_subplot(3, 3, 5)
            im1 = plt.imshow((pos_pred[0, 0, :, :].clone().detach().cpu().numpy()))
            ax1.set_title("LAYER BK - 0")
            f5.colorbar(im1, ax=ax1)
            ax2 = f5.add_subplot(3, 3, 6)
            im2 = plt.imshow((pos_pred[0, 1, :, :].clone().detach().cpu().numpy()))
            ax2.set_title("LAYER OBJ - 1")
            f5.colorbar(im2, ax=ax2)
            ax3 = f5.add_subplot(3, 3, 7)
            im3 = plt.imshow((pos_pred[0, 2, :, :].clone().detach().cpu().numpy()))
            ax3.set_title("LAYER TARGET - 2")
            f5.colorbar(im3, ax=ax3)
            f5.add_subplot(3, 3, 8)
            plt.imshow((fs1[0, 0, :, :].clone().detach().cpu().numpy()))
            f5.add_subplot(3, 3, 9)
            plt.imshow((fs2[0, 0, :, :].clone().detach().cpu().numpy()))

            plt.show()

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
                'pos': pos_pred,
                # 'cos': cos_pred,
                # 'sin': sin_pred,
                'angle': res_angle,
                'width': w_GT
            },
            'pretrained': {
                # 'few_shot': fs_out,
                # 'grasp': res_before
            }
        }
class PROVA_WIDTH_SEG(nn.Module):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """

    def __init__(self, args, position_model, input_channels=3, vis=False, loss=0):
        super(PROVA_WIDTH_SEG, self).__init__()

        self.position_model = position_model

        self.n_classes = 16
        self.input_size = 224

        bilinear = True
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.n_classes)

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

    def compute_loss(self, xc, yc, yc_grasp, rgb_x, rgb_y, support_x, support_y, s_seed, dev):
        # SET TENSORS
        y_pos, y_cos, y_sin, y_width = yc
        y_pos = y_pos.float()
        y_cos = y_cos.float()
        y_sin = y_sin.float()
        y_width = y_width.float()

        y_pos_grasp, y_cos_grasp, y_sin_grasp, y_width_grasp = yc_grasp
        y_pos_grasp = y_pos_grasp.float()
        y_cos_grasp = y_cos_grasp.float()
        y_sin_grasp = y_sin_grasp.float()
        y_width_grasp = y_width_grasp.float()

        xc = xc.float()
        rgb_x = rgb_x.float()
        support_x = support_x.float()
        support_y = support_y.float()

        # GET POSITION
        with torch.no_grad():
            pos_pred, fs1, fs2 = self.position_model(xc, rgb_x, support_x, support_y, s_seed)

        #CREATE FINAL HEATMAP
        final_heatmap = pos_pred[:,2,:,:].clone()
        final_heatmap += pos_pred[:,1,:,:].clone()
        final_heatmap += fs1[:,0,:,:].clone()
        final_heatmap -= pos_pred[:,0,:,:].clone()
        final_heatmap -= fs2[:,0,:,:].clone()

        #GET POSITION AND CROPPED IMAGE
        max_indexes = []
        new_input_tensor = []
        crop_dim = int(self.input_size/2)
        for k in range(y_pos.shape[0]):
            img = final_heatmap[k, :, :]
            local_max = torch.argmax(img.clone())
            local_max = [local_max.cpu()/y_pos.shape[1], local_max.cpu()%y_pos.shape[1]]
            top = max(0, local_max[0]-crop_dim)
            left = max(0,local_max[1]-crop_dim)
            top = min(top, y_pos.shape[1]-self.input_size-1)
            left = min(left, y_pos.shape[1]-self.input_size-1)
            top = int(top)
            left = int(left)
            #new_input_tensor.append(crop(torch.cat((xc[k,:,:,:].unsqueeze(0), fs1[k,:,:,:].unsqueeze(0), fs2[k,:,:,:].unsqueeze(0)), dim=1), top, left, self.input_size, self.input_size))
            #new_input_tensor.append(crop(torch.cat((xc[k,:,:,:].unsqueeze(0), pos_pred[k,1,:,:].unsqueeze(0).unsqueeze(0), pos_pred[k,2,:,:].unsqueeze(0).unsqueeze(0)), dim=1), top, left, self.input_size, self.input_size))
            new_input_tensor.append(torch.cat((xc[k,:,:,:].unsqueeze(0), pos_pred[k,0,:,:].unsqueeze(0).unsqueeze(0), pos_pred[k,1,:,:].unsqueeze(0).unsqueeze(0)), dim=1))
            max_indexes.append(local_max)

        max_indexes = np.array(max_indexes)  # per ogni batch mi trovo il punto di max della heatmap pos
        new_input_tensor = torch.cat(new_input_tensor, dim=0)

        # ANGEL INFERENCE
        width_pred = self(new_input_tensor)

        #COMPUTE ANGLE GT
        # Denormalize
        y_sin_grasp = y_sin_grasp * 2 + (-1)
        y_cos_grasp = y_cos_grasp * 2 + (-1)
        angle_gt = (torch.atan2(y_sin_grasp, y_cos_grasp) / 2.0)
        angle_gt = (angle_gt + math.pi / 2) / math.pi

        width_gt = torch.floor(y_width_grasp * 15)
        #width_gt[width_gt==18] = 17

        cos_GT = torch.empty(y_cos_grasp.shape[0], device=dev)
        sin_GT = torch.empty(y_sin_grasp.shape[0], device=dev)
        w_GT = torch.empty(y_width_grasp.shape[0], device=dev)
        res_angle = torch.empty(angle_gt.shape[0], device=dev)

        for k in range(y_cos_grasp.shape[0]):
            cos_GT[k] = y_cos_grasp[k, int(max_indexes[k, 0]), int(max_indexes[k, 1])]
            sin_GT[k] = y_sin_grasp[k, int(max_indexes[k, 0]), int(max_indexes[k, 1])]
            w_GT[k] = y_width_grasp[k, int(max_indexes[k, 0]), int(max_indexes[k, 1])]
            res_angle[k] = angle_gt[k, int(max_indexes[k, 0]), int(max_indexes[k, 1])]

        # angle creation and normalization
        # pos_pred = y_pos
        # width_pred = y_width

        # print("GT SHAPE", angle_gt.shape)
        w = torch.tensor([0.1,2.0,2.0,2.0,2.0,2.0,2.0,1.5,1.0,1.0,1.0,1.0,0.8,0.8,0.8,0.3]).float().cuda()
        width_loss_grasp = F.cross_entropy(width_pred.float(), width_gt.long(), weight=w)
        # angle_loss_grasp = F.mse_loss(angle, angle_gt_vec)
        # pos_loss = F.mse_loss(pos_pred, y_pos, reduction='sum')
        #pos_mask = y_pos.clone()
        #pos_mask[pos_mask == 1] = 2
        #pos_mask[pos_mask == 0.25] = 1
        # print(pos_pred.shape)
        # print(pos_mask.shape)
        #pos_loss = F.cross_entropy(pos_pred, pos_mask.long())

        #res_angle = torch.argmax(angle_pred, axis=1) / 18 + 1 / 36

        loss = width_loss_grasp

        if self.counter >= (23 * 0) and self.counter % 230 == 1:
            f5 = plt.figure(5)
            f5.suptitle("Images")
            f5.add_subplot(3, 3, 1)
            plt.imshow((width_gt[0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 3, 2)
            plt.imshow((xc[0, 0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 3, 3)
            plt.imshow(torch.argmax(pos_pred.clone(), dim=1).detach().cpu().numpy()[0, :, :])
            f5.add_subplot(3, 3, 4)
            plt.imshow(torch.argmax(width_pred.clone(), dim=1).detach().cpu().numpy()[0, :, :])
            ax1 = f5.add_subplot(3, 3, 5)
            im1 = plt.imshow((width_pred[0, 2, :, :].clone().detach().cpu().numpy()))
            ax1.set_title("LAYER BK - 0")
            f5.colorbar(im1, ax=ax1)
            ax2 = f5.add_subplot(3, 3, 6)
            im2 = plt.imshow((width_pred[0, 6, :, :].clone().detach().cpu().numpy()))
            ax2.set_title("LAYER OBJ - 1")
            f5.colorbar(im2, ax=ax2)
            ax3 = f5.add_subplot(3, 3, 7)
            im3 = plt.imshow((width_pred[0, 12, :, :].clone().detach().cpu().numpy()))
            ax3.set_title("LAYER TARGET - 2")
            f5.colorbar(im3, ax=ax3)
            ax4 = f5.add_subplot(3, 3, 8)
            im4 = plt.imshow((width_pred[0, 13, :, :].clone().detach().cpu().numpy()))
            f5.colorbar(im4, ax=ax4)
            ax5 =f5.add_subplot(3, 3, 9)
            im5 = plt.imshow((width_pred[0, 15, :, :].clone().detach().cpu().numpy()))
            f5.colorbar(im5, ax=ax5)
            path = "/home/barcellona/workspace/git_repo/FSGGCNN/saved_images/"
            f5.savefig(path + str(self.counter) + "iteration_image.png")
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
                'pos': pos_pred,
                # 'cos': cos_pred,
                # 'sin': sin_pred,
                'angle': res_angle,
                'width': w_GT
            },
            'pretrained': {
                # 'few_shot': fs_out,
                # 'grasp': res_before
            }
        }

class PROVA_ANGOLO_SEG(nn.Module):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """

    def __init__(self, args, position_model, input_channels=3, vis=False, loss=0):
        super(PROVA_ANGOLO_SEG, self).__init__()

        self.position_model = position_model

        self.n_classes = 18
        self.input_size = 224

        bilinear = True
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.n_classes)

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

    def compute_loss(self, xc, yc, yc_grasp, rgb_x, rgb_y, support_x, support_y, s_seed, dev):
        # SET TENSORS
        y_pos, y_cos, y_sin, y_width = yc
        y_pos = y_pos.float()
        y_cos = y_cos.float()
        y_sin = y_sin.float()
        y_width = y_width.float()

        y_pos_grasp, y_cos_grasp, y_sin_grasp, y_width_grasp = yc_grasp
        y_pos_grasp = y_pos_grasp.float()
        y_cos_grasp = y_cos_grasp.float()
        y_sin_grasp = y_sin_grasp.float()
        y_width_grasp = y_width_grasp.float()

        xc = xc.float()
        rgb_x = rgb_x.float()
        support_x = support_x.float()
        support_y = support_y.float()

        # GET POSITION
        with torch.no_grad():
            pos_pred, fs1, fs2 = self.position_model(xc, rgb_x, support_x, support_y, s_seed)

        #CREATE FINAL HEATMAP
        final_heatmap = pos_pred[:,2,:,:].clone()
        final_heatmap += pos_pred[:,1,:,:].clone()
        final_heatmap += fs1[:,0,:,:].clone()
        final_heatmap -= pos_pred[:,0,:,:].clone()
        final_heatmap -= fs2[:,0,:,:].clone()

        #GET POSITION AND CROPPED IMAGE
        max_indexes = []
        new_input_tensor = []
        crop_dim = int(self.input_size/2)
        for k in range(y_pos.shape[0]):
            img = final_heatmap[k, :, :]
            local_max = torch.argmax(img.clone())
            local_max = [local_max.cpu()/y_pos.shape[1], local_max.cpu()%y_pos.shape[1]]
            top = max(0, local_max[0]-crop_dim)
            left = max(0,local_max[1]-crop_dim)
            top = min(top, y_pos.shape[1]-self.input_size-1)
            left = min(left, y_pos.shape[1]-self.input_size-1)
            top = int(top)
            left = int(left)
            #new_input_tensor.append(crop(torch.cat((xc[k,:,:,:].unsqueeze(0), fs1[k,:,:,:].unsqueeze(0), fs2[k,:,:,:].unsqueeze(0)), dim=1), top, left, self.input_size, self.input_size))
            #new_input_tensor.append(crop(torch.cat((xc[k,:,:,:].unsqueeze(0), pos_pred[k,1,:,:].unsqueeze(0).unsqueeze(0), pos_pred[k,2,:,:].unsqueeze(0).unsqueeze(0)), dim=1), top, left, self.input_size, self.input_size))
            new_input_tensor.append(torch.cat((xc[k,:,:,:].unsqueeze(0), pos_pred[k,0,:,:].unsqueeze(0).unsqueeze(0), pos_pred[k,1,:,:].unsqueeze(0).unsqueeze(0)), dim=1))
            max_indexes.append(local_max)

        max_indexes = np.array(max_indexes)  # per ogni batch mi trovo il punto di max della heatmap pos
        new_input_tensor = torch.cat(new_input_tensor, dim=0)

        # ANGEL INFERENCE
        angle_pred = self(new_input_tensor)

        #COMPUTE ANGLE GT
        # Denormalize
        y_sin_grasp = y_sin_grasp * 2 + (-1)
        y_cos_grasp = y_cos_grasp * 2 + (-1)

        angle_gt = (torch.atan2(y_sin_grasp, y_cos_grasp) / 2.0)
        angle_gt = (angle_gt + math.pi / 2) / math.pi
        angle_gt = torch.floor(angle_gt * 18)
        angle_gt[angle_gt==18] = 17


        cos_GT = torch.empty(y_cos_grasp.shape[0], device=dev)
        sin_GT = torch.empty(y_sin_grasp.shape[0], device=dev)
        w_GT = torch.empty(y_width_grasp.shape[0], device=dev)
        res_angle = torch.empty(angle_gt.shape[0], device=dev)

        for k in range(y_cos_grasp.shape[0]):
            cos_GT[k] = y_cos_grasp[k, int(max_indexes[k, 0]), int(max_indexes[k, 1])]
            sin_GT[k] = y_sin_grasp[k, int(max_indexes[k, 0]), int(max_indexes[k, 1])]
            w_GT[k] = y_width_grasp[k, int(max_indexes[k, 0]), int(max_indexes[k, 1])]
            res_angle[k] = angle_gt[k, int(max_indexes[k, 0]), int(max_indexes[k, 1])]


        # angle creation and normalization
        '''
        angle_gt = (torch.atan2(sin_GT, cos_GT) / 2.0)
        angle_gt = (angle_gt + math.pi / 2) / math.pi
        angle_gt_vec = torch.empty((y_cos_grasp.shape[0], 18), device=dev).float()
        angle_gt_vec[:, :] = 0
        index_gt = angle_gt * 18 - 1 / 36
        index_gt = index_gt.long()

        for k in range(y_cos_grasp.shape[0]):
            angle_gt_vec[k, index_gt[k]] = 1
        '''



        # pos_pred = y_pos
        # width_pred = y_width

        # print("GT SHAPE", angle_gt.shape)
        # angle_loss_grasp = F.mse_loss(angle_gt, angle[:])
        # angle_loss_grasp = F.cross_entropy(angle.float(), angle_gt_vec.long())
        w = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.2,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]).float().cuda()
        angle_loss_grasp = F.cross_entropy(angle_pred.float(), angle_gt.long(), weight=w)
        # angle_loss_grasp = F.mse_loss(angle, angle_gt_vec)
        # pos_loss = F.mse_loss(pos_pred, y_pos, reduction='sum')
        #pos_mask = y_pos.clone()
        #pos_mask[pos_mask == 1] = 2
        #pos_mask[pos_mask == 0.25] = 1
        # print(pos_pred.shape)
        # print(pos_mask.shape)
        #pos_loss = F.cross_entropy(pos_pred, pos_mask.long())

        #res_angle = torch.argmax(angle_pred, axis=1) / 18 + 1 / 36

        loss = angle_loss_grasp

        if self.counter >= (23 * 0) and self.counter % 230 == 1:
            f5 = plt.figure(5)
            f5.suptitle("Images")
            f5.add_subplot(3, 3, 1)
            plt.imshow((angle_gt[0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 3, 2)
            plt.imshow((xc[0, 0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 3, 3)
            plt.imshow(torch.argmax(pos_pred.clone(), dim=1).detach().cpu().numpy()[0, :, :])
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
            path = "/home/barcellona/workspace/git_repo/FSGGCNN/saved_images/"
            f5.savefig(path + str(self.counter) + "iteration_image.png")
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
                'pos': pos_pred,
                # 'cos': cos_pred,
                # 'sin': sin_pred,
                'angle': res_angle,
                'width': w_GT
            },
            'pretrained': {
                # 'few_shot': fs_out,
                # 'grasp': res_before
            }
        }

class PROVA(nn.Module):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """
    def __init__(self, args, input_channels=3, vis=False, loss=0):
        super(PROVA, self).__init__()

        self.few_shot_model = AsgnetModel(args)

        # FREEZE ASGNET
        for param in self.few_shot_model.parameters():
            param.requires_grad = False

        filter_sizes = [3, 8, 16, 32]
        kernel_sizes = [3, 3, 3, 3]
        strides = [1, 1, 1, 1]
        paddings = [1, 1, 1, 1]
        layer_sizes = [50, 18]
        """
        self.cnn_funnel = nn.Sequential()
        self.intermediate = []

        for i in range(len(kernel_sizes) - 1):
            self.cnn_funnel.add_module("conv2D_{}".format(i),
                                       nn.Conv2d(filter_sizes[i], filter_sizes[i + 1], kernel_sizes[i],
                                                 stride=strides[i], padding=paddings[i]))
            self.cnn_funnel.add_module("batch_{}".format(i), nn.BatchNorm2d(filter_sizes[i + 1]))
            self.cnn_funnel.add_module("relu_{}".format(i), nn.ReLU(inplace=True))
            #self.cnn_funnel.add_module("dropout_{}".format(i), nn.Dropout2d(p=0.1))
            self.cnn_funnel.add_module("max_pool_{}".format(i), nn.MaxPool2d(2, 2))

        #self.dense = MLP(layer_sizes)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

        """
        n_classes = 3
        bilinear = True
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.counter = 0


    def forward(self, depth_x, rgb_x, support_x, support_y, s_seed):
        self.counter += 1
        fs_features, fs_out = self.few_shot_model(rgb_x, s_x=support_x, s_y=support_y, s_seed=s_seed)

        fs = fs_out.clone()
        #print("original: ", fs_features[0].shape)
        #print("reduced: ", fs_features[1].shape)

        fs_out_1 = F.interpolate(fs_out[:,1,:,:].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
        fs_out_2 = F.interpolate(fs_out[:, 0, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
        depth_x = F.interpolate(depth_x[:, 0, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)

        fs_out = torch.cat((depth_x, fs_out_1.detach(), fs_out_2.detach()), 1).cuda()
        fs_out_soft = (fs_out)

        #print(fs_features[1][0,0,0,0])
        #print(fs_out_1[0,0,0,0])

        #fs_features[1][:,:] = fs_features[1][:,:] * fs_out_1
        #print(fs_features[1][0, 0, 0, 0])
        #out = fs_features[1][:,:]
        #out = torch.cat((fs_features[1], fs_out_soft[:,1,:,:].unsqueeze(1).detach()), 1).cuda()

        #out = self.cnn_funnel(fs_out)
        #out = self.up(out)
        #

        x1 = self.inc(fs_out)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out_0 = F.interpolate(logits[:, 0, :, :].unsqueeze(1), size=(721, 721), mode='bilinear', align_corners=False)
        out_1 = F.interpolate(logits[:, 1, :, :].unsqueeze(1), size=(721, 721), mode='bilinear', align_corners=False)
        out_2 = F.interpolate(logits[:, 2, :, :].unsqueeze(1), size=(721, 721), mode='bilinear', align_corners=False)
        out = torch.cat((out_0, out_1, out_2), 1).cuda()
        fs_hmap_1 = fs[:, 1, :, :].unsqueeze(1)
        fs_hmap_2 = fs[:, 0, :, :].unsqueeze(1)
        """
        y = out.clone()
        print("AFTER CAT", out.shape)
        out = self.cnn_funnel(fs_out)
        print("PRE FLATTEN", out.shape)

        if self.counter % 16 == 0 and self.counter >= (18 * 0) and False:
            f4 = plt.figure(4)
            f4.suptitle("Internal Representation")
            for k, layer in enumerate(self.cnn_funnel):
                y = layer(y)
                if k % 4 == 3:
                    for j in range(4):
                        f4.add_subplot(4, 4, (int(k / 4) )*4 + j + 1)
                        img = plt.imshow((y[0, j, :, :].detach().cpu().numpy()))
                        f4.colorbar(img)


            f2 = plt.figure(2)
            f2.suptitle("Images")
            f2.add_subplot(2, 1, 1)
            img = plt.imshow((fs_out[0, 0, :, :].detach().cpu().numpy()))
            f2.colorbar(img)
            f2.add_subplot(2, 1, 2)
            img = plt.imshow((fs_out[0, 1, :, :].detach().cpu().numpy()))
            f2.colorbar(img)

            f1 = plt.figure(1)
            f1.suptitle("Images")
            f1.add_subplot(3, 3, 1)
            plt.imshow((rgb_x.permute(2, 3, 1, 0).detach().cpu().numpy()[:, :, :, 0]).astype(np.uint8)*0.225-0.5)
            f1.add_subplot(3, 3, 2)
            img = plt.imshow((fs_out_soft[0, 0, :, :].detach().cpu().numpy()))
            f1.colorbar(img)
            f1.add_subplot(3, 3, 3)
            img = plt.imshow((fs_out_soft[0, 1, :, :].detach().cpu().numpy()))
            f1.colorbar(img)

            for k, i in enumerate([10, 50, 100, 150, 200, 256]):
                f1.add_subplot(3, 3, k + 4)
                img = plt.imshow(fs_features[0][0, i, :, :].detach().cpu().numpy())
                f1.colorbar(img)

            f3 = plt.figure(3)
            f3.suptitle("Images")
            for k, i in enumerate([0,1,2,3]):
                f3.add_subplot(3, 3, k+1)
                plt.imshow(out[0, i, :, :].detach().cpu().numpy())

        out = torch.flatten(out, 1)
        print("POST FLATTEN",out.shape)
        out = self.dense(out)
        print("OUTPUT SIZE", out.shape)
       
        """
        return out, fs_hmap_1.detach(), fs_hmap_2.detach()

    def compute_loss(self, xc, yc, yc_grasp, rgb_x, rgb_y, support_x, support_y, s_seed, dev):
        y_pos, y_cos, y_sin, y_width = yc

        y_pos = y_pos.float()
        y_cos = y_cos.float()
        y_sin = y_sin.float()
        y_width = y_width.float()

        y_pos_grasp, y_cos_grasp, y_sin_grasp, y_width_grasp = yc_grasp

        y_pos_grasp = y_pos_grasp.float()
        y_cos_grasp = y_cos_grasp.float()
        y_sin_grasp = y_sin_grasp.float()
        y_width_grasp = y_width_grasp.float()

        xc = xc.float()
        rgb_x = rgb_x.float()
        support_x = support_x.float()
        support_y = support_y.float()

        pos_pred, fs1, fs2 = self(xc, rgb_x, support_x, support_y, s_seed)

        y_pos_clone = torch.argmax(pos_pred.clone(), dim=1).detach().cpu().numpy()
        max_indexes = []
        for k in range(y_pos.shape[0]):
            img = y_pos_clone[k, :, :]
            img = gaussian(img, 2.0, preserve_range=True)
            local_max = peak_local_max(img, min_distance=20, threshold_abs=0.0, num_peaks=1)
            if len(local_max) < 1:
                #print("No peaks")
                #print(local_max)
                h = 721
                w = 721
                local_max = np.expand_dims(np.array((int(h / 2), int(w / 2)), np.int), axis=0)
                #print(local_max)
                #print("Changed peaks")
            max_indexes.append(local_max)

        max_indexes = np.array(max_indexes)  # per ogni batch mi trovo il punto di max della heatmap pos

        # Denormalize
        y_sin_grasp = y_sin_grasp * 2 + (-1)
        y_cos_grasp = y_cos_grasp * 2 + (-1)

        cos_GT = torch.empty(y_cos_grasp.shape[0], device=dev)
        sin_GT = torch.empty(y_sin_grasp.shape[0], device=dev)
        w_GT = torch.empty(y_width_grasp.shape[0], device=dev)

        for k in range(y_cos_grasp.shape[0]):
            cos_GT[k] = y_cos_grasp[k, max_indexes[k, 0, 0], max_indexes[k, 0, 1]]
            sin_GT[k] = y_sin_grasp[k, max_indexes[k, 0, 0], max_indexes[k, 0, 1]]
            w_GT[k] = y_width_grasp[k, max_indexes[k, 0, 0], max_indexes[k, 0, 1]]

        # angle creation and normalization
        angle_gt = (torch.atan2(sin_GT, cos_GT) / 2.0)
        angle_gt = (angle_gt + math.pi / 2) / math.pi
        angle_gt_vec = torch.empty((y_cos_grasp.shape[0],18), device=dev).float()
        angle_gt_vec[:,:] = 0
        index_gt = angle_gt*18 - 1/36
        index_gt = index_gt.long()

        for k in range(y_cos_grasp.shape[0]):
            angle_gt_vec[k,index_gt[k]] = 1

        #pos_pred = y_pos
        width_pred = y_width

        #print("GT SHAPE", angle_gt.shape)
        #angle_loss_grasp = F.mse_loss(angle_gt, angle[:])
        #angle_loss_grasp = F.cross_entropy(angle.float(), angle_gt_vec.long())
        #angle_loss_grasp = F.cross_entropy(angle,index_gt)
        #angle_loss_grasp = F.mse_loss(angle, angle_gt_vec)
        #pos_loss = F.mse_loss(pos_pred, y_pos, reduction='sum')
        pos_mask = y_pos.clone()
        pos_mask[pos_mask == 1] = 2
        pos_mask[pos_mask == 0.25] = 1
        #print(pos_pred.shape)
        #print(pos_mask.shape)
        pos_loss = F.cross_entropy(pos_pred, pos_mask.long())

        res_angle = torch.argmax(angle_gt_vec, axis=1)/18 + 1/36
        #angle = angle.squeeze(1)
        #print("angle_gt {} , angle_pred {}: ".format(angle_gt,angle_pred))
        #print("angle loss: ", angle_loss_grasp)

        #width_loss_grasp = F.mse_loss(width_pred[:,0], w_GT)

        #loss = p_loss + angle_loss_grasp + width_loss_grasp
        loss = pos_loss

        if self.counter >= (18 * 0) and self.counter%180 == 1 or True:
            f5 = plt.figure(5)
            f5.suptitle("Images")
            f5.add_subplot(3, 3, 1)
            plt.imshow((y_pos[0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 3, 2)
            plt.imshow((xc[0, 0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 3, 3)
            plt.imshow(torch.argmax(pos_pred.clone(), dim=1).detach().cpu().numpy()[0, :, :])
            f5.add_subplot(3, 3, 4)
            plt.imshow(pos_mask[0, :, :].detach().cpu().numpy())
            ax1 =f5.add_subplot(3, 3, 5)
            im1 = plt.imshow((pos_pred[0, 0, :, :].clone().detach().cpu().numpy()))
            ax1.set_title("LAYER BK - 0")
            f5.colorbar(im1, ax=ax1)
            ax2 = f5.add_subplot(3, 3, 6)
            im2 = plt.imshow((pos_pred[0, 1, :, :].clone().detach().cpu().numpy()))
            ax2.set_title("LAYER OBJ - 1")
            f5.colorbar(im2, ax=ax2)
            ax3 = f5.add_subplot(3, 3, 7)
            im3 = plt.imshow((pos_pred[0, 2, :, :].clone().detach().cpu().numpy()))
            ax3.set_title("LAYER TARGET - 2")
            f5.colorbar(im3, ax=ax3)
            f5.add_subplot(3, 3, 8)
            plt.imshow((fs1[0, 0, :, :].clone().detach().cpu().numpy()))
            f5.add_subplot(3, 3, 9)
            plt.imshow((fs2[0, 0, :, :].clone().detach().cpu().numpy()))

            plt.show()

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
                'pos': pos_pred,
                #'cos': cos_pred,
                #'sin': sin_pred,
                'angle': res_angle,
                'width': w_GT
            },
            'pretrained':{
                #'few_shot': fs_out,
                #'grasp': res_before
            }
        }

class GGCNN_ASGNET(nn.Module):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """
    def __init__(self, args, input_channels=3, vis=False, loss=0):
        super(GGCNN_ASGNET, self).__init__()
      
        #INSTANCIATE MODELS
        self.few_shot_model = AsgnetModel(args)
        self.model_ggcnn = GGCNN()

        #FREEZE GGCNN
        for param in self.model_ggcnn.parameters():
            param.requires_grad = False

        #FREEZE ASGNET
        for param in self.few_shot_model.parameters():
            param.requires_grad = False


        #CREATE 4 CONV LAYERS
        self.pos_fusion =FusionLayer()
        #self.cos_fusion =FusionLayer()
        #self.sin_fusion =FusionLayer()
        self.width_fusion = FusionLayer()
        self.float()

        self.vis = vis
    
    def forward(self, depth_x, rgb_x, support_x, support_y, s_seed):

        #depth_x = depth_x.float()
        #rgb_x = rgb_x.float()
        #support_x = support_x.float()
        #support_y = support_y.float()

        #FEATURE DA DEPTH
        depth_features, pos, cos, sin, width = self.model_ggcnn(depth_x)

        #depth_features -> (batch, 32, 265, 265)

        #FEATURE DA FEW SHOT
        fs_features, fs_out = self.few_shot_model(rgb_x, s_x=support_x, s_y=support_y, s_seed=s_seed)

        fs_out_soft = normalize_fs_out(fs_out)

        """
        if self.vis:
            print("Size depth:",depth_x.shape)
            print("Size rgb:",rgb_x.shape)

            print("Features depth")
            print(depth_features.shape)   
            print("Features fs")
            print(fs_features.shape)

            print("Heatmap pos")
            print(pos.shape)
            print(pos[0,0,:,:].shape)
            print("Heatmap fs")
            print(fs_out.shape)
        """
        res_before = [(pos).clone(), (cos).clone(), (sin).clone(), (width).clone() ]

        pos_out = F.interpolate(pos, size=(fs_out[0].shape[2], fs_out.shape[3]), mode='bilinear', align_corners=True)
        cos_out = F.interpolate(cos, size=(fs_out[0].shape[2], fs_out.shape[3]), mode='bilinear', align_corners=True)
        sin_out = F.interpolate(sin, size=(fs_out[0].shape[2], fs_out.shape[3]), mode='bilinear', align_corners=True)
        width_out = F.interpolate(width, size=(fs_out[0].shape[2], fs_out.shape[3]), mode='bilinear', align_corners=True)

        pos_cat = torch.cat((pos_out, fs_out_soft.detach()), 1).cuda()
        #cos_cat = torch.cat((cos_out, fs_out.detach()), 1).cuda()
        #sin_cat = torch.cat((sin_out, fs_out.detach()), 1).cuda()
        width_cat = torch.cat((width_out, fs_out_soft.detach()), 1).cuda()

        #First fusion
        pos_out = self.pos_fusion(pos_cat)
        #print(self.pos_fusion.fusion[0].weight.data)
        #print( self.pos_fusion.weight)
     
        #Second fusion
        #cos_out = self.cos_fusion(cos_cat)
      
        #Third fusion
        #sin_out = self.sin_fusion(sin_cat)
     
        #Fourth fusion
        width_out = self.width_fusion(width_cat)

        """
        if self.vis:
            print("Heatmap concat")
            print(pos_cat.shape)
            print("Final Layer pose")
            print(pos_out.shape)
        """
        if self.vis:
            return pos_out, cos_out, sin_out, width_out, fs_out, res_before
        else:   
            return pos_out, cos_out, sin_out, width_out,

    def compute_loss(self, xc, yc, yc_grasp, rgb_x, rgb_y, support_x, support_y, s_seed, device):
        y_pos, y_cos, y_sin, y_width = yc

        y_pos = y_pos.float()
        y_cos = y_cos.float()
        y_sin = y_sin.float()
        y_width = y_width.float()

        y_pos_grasp, y_cos_grasp, y_sin_grasp, y_width_grasp = yc_grasp

        y_pos_grasp = y_pos_grasp.float()
        y_cos_grasp = y_cos_grasp.float()
        y_sin_grasp = y_sin_grasp.float()
        y_width_grasp = y_width_grasp.float()

        xc = xc.float()
        rgb_x = rgb_x.float()
        support_x = support_x.float()
        support_y = support_y.float()

        if self.vis:
            pos_pred, cos_pred, sin_pred, width_pred, fs_out, res_before = self(xc, rgb_x, support_x, support_y, s_seed)
        else:
            pos_pred, cos_pred, sin_pred, width_pred = self(xc, rgb_x, support_x, support_y, s_seed)

        pos_pred = pos_pred.squeeze(1)
        cos_pred = cos_pred.squeeze(1)
        sin_pred = sin_pred.squeeze(1)
        width_pred = width_pred.squeeze(1)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos_grasp)
        sin_loss = F.mse_loss(sin_pred, y_sin_grasp)
        width_loss = F.mse_loss(width_pred, y_width_grasp)
        """
        if self.vis:
            print("y pos",y_pos.shape)
            print("pos_pred",pos_pred.shape)
            print("Loss:", p_loss)
        """
        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'info': {
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            },
            'pretrained':{
                'few_shot': fs_out,
                'grasp': res_before
            }
        }
