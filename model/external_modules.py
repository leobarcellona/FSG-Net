import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as T

from model.model_asgnet.asgnet_features import AsgnetModel
from model.model_ggcnn.ggcnn_features import GGCNN

from model.deeplab.DeepLabLayers import DeepLabHeadV3Plus, ASPP

import math

from skimage.feature import peak_local_max
import numpy as np
from skimage.filters import gaussian

class CNN_EncDec(nn.Module):
    def __init__(self, filter_sizes, kernel_sizes, strides, paddings, max_pool, input_channels=3):
        super().__init__()

        self.enc = nn.Sequential()

        for i in range(len(kernel_sizes)-1):
            self.enc.add_module("conv2D_{}".format(i),nn.Conv2d(filter_sizes[i], filter_sizes[i+1], kernel_sizes[i], stride=strides[i], padding=paddings[i]))
            self.enc.add_module("batch_{}".format(i), nn.BatchNorm2d(filter_sizes[i+1]))
            self.enc.add_module("relu_{}".format(i),nn.ReLU(inplace=True))
            self.enc.add_module("dropout_{}".format(i),nn.Dropout2d(p=0.1))
            if max_pool[i]:
                self.enc.add_module("max_pool_{}".format(i),nn.MaxPool2d(2, 2))

        self.dec = nn.Sequential()
        self.dec.add_module("pointwise_conv",nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=1))

        for j in range(len(kernel_sizes)):
            i = -j -1
            #self.dec.add_module("convt_{}".format(i),nn.ConvTranspose2d(filter_sizes[i], filter_sizes[i+1], kernel_sizes[i], stride=strides[i], padding=1, output_padding=1))
            self.dec.add_module("up_{}".format(i),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.dec.add_module("conv_{}".format(i),nn.Conv2d(filter_sizes[i], filter_sizes[i-1], kernel_sizes[i-1], stride=1, padding=0))
            self.dec.add_module("prelu_{}".format(i),nn.PReLU())
            self.dec.add_module("batchnorm_{}".format(i),nn.BatchNorm2d(filter_sizes[i+2]))
        
        self.dec.add_module("up_last",nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))  

    def forward(self, x):

        out = self.enc(x)

        return out


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

        fs_out_1 = F.interpolate(fs_out[:, 1, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
        fs_out_2 = F.interpolate(fs_out[:, 0, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
        depth_x = F.interpolate(depth_x[:, 0, :, :].unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)

        fs_out = torch.cat((depth_x, fs_out_1.detach(), fs_out_2.detach()), 1).cuda()
        fs_out_soft = (fs_out)

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
        return out

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

        pos_pred = self(xc, rgb_x, support_x, support_y, s_seed)

        y_pos_clone = torch.argmax(pos_pred.clone(), dim=1).detach().cpu().numpy()
        max_indexes = []
        for k in range(y_pos.shape[0]):
            img = y_pos_clone[k, :, :]
            img = gaussian(img, 2.0, preserve_range=True)
            local_max = peak_local_max(img, min_distance=20, threshold_abs=0.0, num_peaks=1)
            if len(local_max) < 1:
                h = 721
                w = 721
                local_max = np.expand_dims(np.array((int(h / 2), int(w / 2)), np.int), axis=0)
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
        angle_gt_vec = torch.empty((y_cos_grasp.shape[0], 18), device=dev).float()
        angle_gt_vec[:, :] = 0
        index_gt = angle_gt * 18 - 1 / 36
        index_gt = index_gt.long()

        for k in range(y_cos_grasp.shape[0]):
            angle_gt_vec[k, index_gt[k]] = 1

        # pos_pred = y_pos
        width_pred = y_width

        # print("GT SHAPE", angle_gt.shape)
        # angle_loss_grasp = F.mse_loss(angle_gt, angle[:])
        # angle_loss_grasp = F.cross_entropy(angle.float(), angle_gt_vec.long())
        # angle_loss_grasp = F.cross_entropy(angle,index_gt)
        # angle_loss_grasp = F.mse_loss(angle, angle_gt_vec)
        # pos_loss = F.mse_loss(pos_pred, y_pos, reduction='sum')
        pos_mask = y_pos.clone()
        pos_mask[pos_mask == 1] = 2
        pos_mask[pos_mask == 0.25] = 1
        # print(pos_pred.shape)
        # print(pos_mask.shape)
        pos_loss = F.cross_entropy(pos_pred, pos_mask.long())

        res_angle = torch.argmax(angle_gt_vec, axis=1) / 18 + 1 / 36
        # angle = angle.squeeze(1)
        # print("angle_gt {} , angle_pred {}: ".format(angle_gt,angle_pred))
        # print("angle loss: ", angle_loss_grasp)

        # width_loss_grasp = F.mse_loss(width_pred[:,0], w_GT)

        # loss = p_loss + angle_loss_grasp + width_loss_grasp
        loss = pos_loss

        if self.counter >= (18 * 0) and self.counter % 180 == 1 and False:
            plt.show()
            f5 = plt.figure(5)
            f5.suptitle("Images")
            f5.add_subplot(3, 2, 1)
            plt.imshow((pos_pred[0, 0, :, :].clone().detach().cpu().numpy()))
            f5.add_subplot(3, 2, 2)
            plt.imshow((y_pos[0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 2, 3)
            plt.imshow((xc[0, 0, :, :].detach().cpu().numpy()))
            f5.add_subplot(3, 2, 4)
            plt.imshow(torch.argmax(pos_pred.clone(), dim=1).detach().cpu().numpy()[0, :, :])
            f5.add_subplot(3, 2, 5)
            plt.imshow(pos_mask[0, :, :].detach().cpu().numpy())

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
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def normalize_fs_out(fs_out):

    batch, ch, h, w = fs_out.shape

    fs_out_view = fs_out.view(batch, ch, h*w)

    fs_out_mean = torch.mean(fs_out_view,dim=2) 
    fs_out_std = torch.std(fs_out_view, dim=2)

    fs_out_mean = fs_out_mean.view(1,batch*ch).repeat(1, h*w).view(batch,ch,h,w)
    fs_out_std = fs_out_std.view(1,batch*ch).repeat(1, h*w).view(batch,ch,h,w)

    fs_out_soft = ((fs_out - fs_out_mean) / fs_out_std) + 0.5

    return fs_out_soft


# CLASSES


class CNN_funnel(nn.Module):
    def __init__(self, filter_sizes, kernel_sizes, strides, paddings, max_pool, input_channels=3):
        super().__init__()

        self.cnn_funnel = nn.Sequential()

        for i in range(len(kernel_sizes)-1):
            self.cnn_funnel.add_module("conv2D_{}".format(i),nn.Conv2d(filter_sizes[i], filter_sizes[i+1], kernel_sizes[i], stride=strides[i], padding=paddings[i]))
            self.cnn_funnel.add_module("batch_{}".format(i), nn.BatchNorm2d(filter_sizes[i+1]))
            self.cnn_funnel.add_module("relu_{}".format(i),nn.ReLU(inplace=True))
            self.cnn_funnel.add_module("dropout_{}".format(i),nn.Dropout2d(p=0.1))
            if max_pool[i]:
                self.cnn_funnel.add_module("max_pool_{}".format(i),nn.MaxPool2d(2, 2))

    def forward(self, x):

        out = self.cnn_funnel(x)

        return out


class MLP(nn.Module):

    def __init__(self, layer_sizes):
        super().__init__()

        self.dense_layers = nn.Sequential()

        for i in range(len(layer_sizes)-2):
            self.dense_layers.add_module("linear_{}".format(i),nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.dense_layers.add_module("batch_{}".format(i), nn.BatchNorm1d(layer_sizes[i+1]))
            self.dense_layers.add_module("relu{}".format(i), nn.ReLU(inplace=True))

        self.dense_layers.add_module("linear_out", nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    def forward(self, x):
        return self.dense_layers(x)


class FeatureLayer(nn.Module):
    def __init__(self, filter_sizes_features, kernel_sizes_features, strides_features, paddings_features, input_channels=3):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels, filter_sizes_features[0], kernel_sizes_features[0], stride=strides_features[0],
                      padding=paddings_features[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(filter_sizes_features[0], filter_sizes_features[1], kernel_sizes_features[1],
                      stride=strides_features[1], padding=paddings_features[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(filter_sizes_features[1], filter_sizes_features[2], kernel_sizes_features[2],
                      stride=strides_features[2], padding=paddings_features[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            #nn.Conv2d(filter_sizes_features[2], filter_sizes_features[3], kernel_sizes_features[3],
                      #stride=strides_features[3], padding=paddings_features[3]),
            #nn.ReLU(inplace=True),

            # nn.Conv2d(filter_sizes[3], 1, kernel_size=2)
        )
        

        for m in self.fusion:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        return self.fusion(x)


class DeepLabModule(nn.Module):
    def __init__(self, filter_sizes_features, kernel_sizes_features, strides_features, paddings_features, input_channels=3, vis=False, aspp_dilate=[12, 24, 36], num_classes = 1):
        super(DeepLabModule, self).__init__()

        high_leve_channels=filter_sizes_features[2]

        self.features_extractor = FeatureLayer(filter_sizes_features, kernel_sizes_features, strides_features, paddings_features, input_channels)

        ll_ch = 16
        hl_ch = 26
        out_ch = 8

        self.project = nn.Sequential(
            nn.Conv2d(input_channels, ll_ch, 1, bias=False),
            nn.BatchNorm2d(ll_ch),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(high_leve_channels, aspp_dilate, output_ch = hl_ch)
        self.classifier = nn.Sequential(
            nn.Conv2d(ll_ch+hl_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, num_classes, 1)
        )

    def forward(self, features):
        feature_low = features.clone()
        #print(feature_low.shape)
        features_high = self.features_extractor(features)
        #print(features_high.shape)
        low_level_feature = self.project(feature_low)
        #print(low_level_feature.shape)
        output_feature = self.aspp(features_high)
        #print(output_feature.shape)
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        #print(output_feature.shape)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_path = './initmodel/resnet50_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    return model

BatchNorm = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=False):
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BatchNorm(64)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(1, 64, stride=2)
            self.bn1 = BatchNorm(64)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm(64)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm(128)
            self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x

class decoder(nn.Module):
    def __init__(self, filter_sizes, kernel_sizes, strides):
        super().__init__()

        self.dec = nn.Sequential()

        for i in range(len(kernel_sizes)-1):
            self.dec.add_module("convt_{}".format(i),nn.ConvTranspose2d(filter_sizes[i], filter_sizes[i+1], kernel_sizes[i], stride=strides[i], padding=1, output_padding=1))

    def forward(self, x):
        return self.dec(x)


