import random
import math
import numpy as np
import numbers
import collections
import cv2

import torch

manual_seed = 123
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
random.seed(manual_seed)


class ComposeGrasp(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, depth, heatmaps):
        for t in self.segtransform:
            depth, heatmaps = t(depth, heatmaps)
        return depth, heatmaps

import time
class ToTensorGrasp(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, depth, heatmaps):

        depth = torch.from_numpy(depth)
        if not isinstance(depth, torch.FloatTensor):
            depth = depth.float()

        for h_index in range(len(heatmaps)):
            heatmaps[h_index] = torch.from_numpy(heatmaps[h_index])
            if not isinstance(depth, torch.FloatTensor):
                heatmaps[h_index] = heatmaps[h_index].float()

        return depth, heatmaps


class NormalizeGrasp(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, depth, heatmaps):

        depth = np.clip((depth - depth.mean()), -1, 1)
        
        return depth, heatmaps





class ResizeGrasp(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        self.size = size

    def __call__(self, depth, heatmaps):

        test_size = self.size
        def find_new_hw(ori_h, ori_w, test_size):
            if ori_h >= ori_w:
                ratio = test_size*1.0 / ori_h
                new_h = test_size
                new_w = int(ori_w * ratio)
            elif ori_w > ori_h:
                ratio = test_size*1.0 / ori_w
                new_h = int(ori_h * ratio)
                new_w = test_size

            if new_h % 8 != 0:
                new_h = (int(new_h /8))*8
            else:
                new_h = new_h
            if new_w % 8 != 0:
                new_w = (int(new_w /8))*8
            else:
                new_w = new_w    
            return new_h, new_w
        new_h, new_w = find_new_hw(depth.shape[0], depth.shape[1], test_size)

        depth_crop = cv2.resize(depth, dsize=(int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
        back_crop_depth = np.zeros((test_size, test_size)) 
        back_crop_depth[:new_h, :new_w] = depth_crop
        depth = back_crop_depth 

        for h_index in range(len(heatmaps)):
            heatmap_crop = cv2.resize(heatmaps[h_index], dsize=(int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
            back_crop_heatmaps = np.zeros((test_size, test_size))
            back_crop_heatmaps[:new_h, :new_w] = heatmap_crop
            heatmaps[h_index] = back_crop_heatmaps


        return depth, heatmaps



class RandScaleGrasp(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label, depth, heatmaps):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        for h_index in range(len(heatmaps)):
            heatmaps[h_index] = cv2.resize(heatmaps[h_index], None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label, depth, heatmaps


class CropGrasp(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        self.size = size
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand' or crop_type=='custom':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, depth, heatmaps):
        h, w = depth.shape
        #print("h",h)
        #print("w",w)

        if h<w:
            smaller_size = self.crop_h
            max_crop_h = h-self.crop_h
            max_crop_w = w-self.crop_h

        else:
            smaller_size = self.crop_w
            max_crop_h = h - self.crop_w
            max_crop_w = w - self.crop_w

        #print("smaller_size", smaller_size)
        #print("ch_size", max_crop_h)
        #print("cw_size", max_crop_w)

        if max_crop_w>0:
            crop_w = np.random.randint(0,max_crop_w)
        else:
            crop_w = 0
        if max_crop_h>0:
            crop_h = np.random.randint(0,max_crop_h)
        else:
            crop_h=0
        if self.crop_type == 'custom':

            depth = depth[:, 280:-280]
            for h_index in range(len(heatmaps)):
                heatmaps[h_index] = heatmaps[h_index][:, 280:-280]

            return depth, heatmaps

        depth = depth[crop_h:crop_h+smaller_size,crop_w:crop_w+smaller_size]
        for h_index in range(len(heatmaps)):
            heatmaps[h_index] = heatmaps[h_index][crop_h:crop_h+smaller_size,crop_w:crop_w+smaller_size]

        return depth, heatmaps


class RandRotateGrasp(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label, depth, heatmaps):
        cost_val = 0
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
            depth = cv2.warpAffine(depth, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=cost_val)
            for h_index in range(len(heatmaps)):
                heatmaps[h_index] = cv2.warpAffine(heatmaps[h_index], matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=cost_val)
        return image, label, depth, heatmaps


class RandomHorizontalFlipGrasp(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, depth, heatmaps):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
            depth = cv2.flip(depth, 1)
            for h_index in range(len(heatmaps)):
                heatmaps[h_index] = cv2.flip(heatmaps[h_index], 1)
        return image, label, depth, heatmaps


class RandomVerticalFlipGrasp(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, depth, heatmaps):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
            depth = cv2.flip(depth, 0)
            for h_index in range(len(heatmaps)):
                heatmaps[h_index] = cv2.flip(heatmaps[h_index], 0)
        return image, label, depth, heatmaps


class RandomGaussianBlurGrasp(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label, depth, heatmaps):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label, depth, heatmaps


class RGB2BGRGrasp(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label, depth, heatmaps):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label, depth, heatmaps


class BGR2RGBGrasp(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label, depth, heatmaps):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label, depth, heatmaps
