

import torch.nn as nn
import torch.nn.functional as F
import torch


from model.model_asgnet.asgnet_features import AsgnetModel 
from model.model_ggcnn.ggcnn_features import GGCNN
from model.external_modules import normalize_fs_out, MLP



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
