import os
import os.path
import cv2
import numpy as np
import glob

from torch.utils.data import Dataset
from util.seed_init import place_seed_points

import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from util.GraspRectangle import GraspRectangles, GraspRectangle, Grasp
from util.dataset_processing import grasp, image
from imageio import imread


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(data_root=None, data_list=None, sub_list=None):
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)      
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')

        # DA AGGIUSTARE: ORDINE OGGETTI
        rgb_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        depth_name = os.path.join(data_root, line_split[2])
        grasp_name = os.path.join(data_root, line_split[3])  # path to rectangular grasps

        item = (rgb_name, label_name, depth_name, grasp_name)

        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        # remove 0 and 255 in the label image
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        # remove those images with small objects
        new_label_class = []
        for c in label_class:
            if c in sub_list:
                tmp_label = np.zeros_like(label)
                target_pix = np.where(label == c)
                tmp_label[target_pix[0], target_pix[1]] = 1

                # proviamo a tenere quelli piccoli?
                if tmp_label.sum() >= 2 * 32 * 32:
                    new_label_class.append(c)
                # new_label_class.append(c)

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)

    print("{} images&label pair after processing! ".format(len(image_label_list)))
    return image_label_list, sub_class_file_list


def read_classes(class_path):
    classes = []
    file = open(class_path, "r")
    for line in file:
        classes.append(int(line))
    file.close()
    return classes


# data_root = "/media/data/Datasets/graspnet/scenes"
class GraspingData(Dataset):
    def __init__(self, shot=1, max_sp=5, data_root=None, data_list=None, data_classes=None,
                 transform=None, transform_shots=None, mode='train', num_all_classes=None, gripper_width=150):

        assert mode in ['train', 'val', 'test']

        self.mode = mode
        self.shot = shot
        self.max_sp = max_sp
        self.data_root = data_root
        self.transform = transform
        self.transform_shots = transform_shots
        self.gripper_width = gripper_width

        # CONTIENE LA LISTA DEGLI ID ORIGINALI DI GRASPNET (IL LABEL=ID+1)
        self.graspnet_training_classes = read_classes(data_classes)

        if (self.mode == 'train') | (self.mode == 'val'):

            self.data_list = []
            self.sub_class_file_list = []

            folders = glob.glob(self.data_root+"/*")
            for lab, folder in enumerate(folders):
                files_rgb = glob.glob(folder+"/*RGB.png")
                if len(files_rgb) < 5:
                    continue

                files_depth = glob.glob(folder + "/*stereo_depth.tiff")
                files_mask = glob.glob(folder + "/*mask.png")
                files_grasps = glob.glob(folder + "/*.txt")
                files_rgb.sort()
                files_depth.sort()
                files_mask.sort()
                files_grasps.sort()
                self.sub_class_file_list.append([])
                for i in range(len(files_rgb)):
                    item = (files_rgb[i], files_mask[i], files_depth[i], files_grasps[i], lab)
                    self.data_list.append(item)
                    self.sub_class_file_list[-1].append(item)


    def _get_crop_attrs(self, idx):
        gtbbs = GraspRectangles.load_from_graspnet_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 1280 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 720 - self.output_size))
        return center, left, top

    """
    def get_gtbb(self, grasp_path, rot=0, zoom=1.0, obj_id=None, n_grasps=100):
        gtbbs = GraspRectangles.load_from_graspnet_file(grasp_path,
                                                        obj_id=obj_id, num_grasps = n_grasps)  # num_grasps = grasps generated per object
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
        return gtbbs
    """
    def get_gtbb(self, grasp_path, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_jacquard_file(grasp_path)
        #c = self.output_size // 2
        #gtbbs.rotate(rot, (c, c))
        #gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def __len__(self):

        return len(self.data_list)

    # RESITUTISCE UN IMMAGINE DI QUERY CON UN SUPPORTO CASUALE
    def __getitem__(self, index):

        # LEGGE L'IMMAGINE E IL LABEL DI TARGET (target skippato per inference)
        label_class = []

        rgb_path, label_path, depth_path, grasp_path, lab = self.data_list[index]  # 4 path to data

        # LEGGE LABEL, RGB, DEPTH, GRASP HEATMAP
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        #plt.imshow(rgb)
        #plt.show()
        rgb = np.float32(rgb)
        #depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        #depth = np.float32(depth)
        #depth = image.DepthImage.from_tiff(depth_path)
        depth = imread(depth_path)
        #depth.normalise()
        #plt.imshow(depth)
        #plt.show()

        #plt.imshow(label)
        #plt.show()

        class_chosen = lab

        # DA PREPARARE LE 4 HEATMAP GT
        # Load the grasps of the selected objects
        bbs = self.get_gtbb(grasp_path, rot=0, zoom=1.0)
        shape = (np.shape(label))
        pos_img_semantic, ang_img_semantic, width_img_semantic = bbs.draw(shape)
        width_img_semantic = np.clip(width_img_semantic, 0.0, self.gripper_width) / self.gripper_width

        pos_hmap = pos_img_semantic
        cos_hmap = np.cos(2 * ang_img_semantic)
        sin_hmap = np.sin(2 * ang_img_semantic)
        widht_hmap = width_img_semantic

        #plt.imshow(pos_hmap)
        #plt.show()
        #plt.imshow(cos_hmap)
        #plt.show()

        # Load the grasps of all objects
        bbs = self.get_gtbb(grasp_path, rot=0, zoom=1.0)
        shape = (np.shape(label))
        pos_img_grasp, ang_img_grasp, width_img_grasp = bbs.draw(shape)
        width_img_grasp = np.clip(width_img_grasp, 0.0, self.gripper_width) / self.gripper_width

        pos_hmap_grasp = pos_img_grasp
        cos_hmap_grasp = np.cos(2 * ang_img_grasp)
        sin_hmap_grasp = np.sin(2 * ang_img_grasp)
        widht_hmap_grasp = width_img_grasp

        #WEIGHTED HEATMAP with ONE and all obejcts
        pos_hmap = pos_img_grasp*0.25 + pos_hmap*0.75
        widht_hmap =  widht_hmap_grasp*0.25 + widht_hmap*0.75


        # TROVA TUTTI I FILE DELLA CLASSE SCELTA
        file_class_chosen = self.sub_class_file_list[lab]
        num_file = len(file_class_chosen)
        #print(num_file)

        # SCEGLIE K IMMAGINI DI SUPPORTO CASUALI (k shot)
        support_rgb_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1, num_file) - 1
            support_rgb_path = rgb_path
            support_label_path = label_path
            while (support_idx in support_idx_list):
                support_idx = random.randint(1, num_file) - 1
                support_rgb_path, support_label_path, _, _, _ = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_rgb_path_list.append(support_rgb_path)
            support_label_path_list.append(support_label_path)

        # CARICA IL VETTORE DI IMMAGINI E LABEL E LI SISTEMA (rende binari come prima)
        support_rgb_list = []
        support_label_list = []
        subcls_list = []
        for k in range(self.shot):
            if (self.mode == 'train'):
                subcls_list.append((class_chosen))
            else:
                subcls_list.append((class_chosen))
                # print(self.sub_val_list.index(class_chosen))
            support_rgb_path = support_rgb_path_list[k]
            support_label_path = support_label_path_list[k]
            support_rgb = cv2.imread(support_rgb_path, cv2.IMREAD_COLOR)
            support_rgb = cv2.cvtColor(support_rgb, cv2.COLOR_BGR2RGB)
            support_rgb = np.float32(support_rgb)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            if support_rgb.shape[0] != support_label.shape[0] or support_rgb.shape[1] != support_label.shape[1]:
                raise (RuntimeError(
                    "Support Image & label shape mismatch: " + support_rgb_path + " " + support_label_path + "\n"))
            support_rgb_list.append(support_rgb)
            support_label_list.append(support_label)
            #plt.imshow(support_rgb)
            #plt.imshow(support_label)
            #plt.show()
        assert len(support_label_list) == self.shot and len(support_rgb_list) == self.shot

        # TRASFORA LE IMMAGINI RGB, I LABEL E LA DEPTH IN TENSORI
        raw_label = label.copy()

        if self.transform is not None:
            rgb_copy = rgb.copy()
            # definiamo una classe transform2 per gestire x argomenti in input
            rgb, label, depth, heatmaps = self.transform(rgb, depth, label, [pos_hmap, cos_hmap, sin_hmap,
                                                                             widht_hmap,pos_hmap_grasp, cos_hmap_grasp,
                                                                             sin_hmap_grasp, widht_hmap_grasp])
            pos_hmap = heatmaps[0]
            cos_hmap = heatmaps[1]
            sin_hmap = heatmaps[2]
            widht_hmap = heatmaps[3]
            pos_hmap_grasp = heatmaps[4]
            cos_hmap_grasp = heatmaps[5]
            sin_hmap_grasp = heatmaps[6]
            widht_hmap_grasp = heatmaps[7]

            for k in range(self.shot):
                support_rgb_list[k], support_label_list[k] = self.transform_shots(support_rgb_list[k],
                                                                                  support_label_list[k])

        # CONCATENA IL SUPPORTO
        s_xs = support_rgb_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        # For every support label, making the corresponding initial sp seed: b x shot x num_sp x 2
        # SPECIFICO PER ASGNet
        init_seed_list = []
        for i in range(0, self.shot):
            mask = (s_y[i, :, :] == 1).float()  # H x W
            init_seed = place_seed_points(mask, down_stride=8, max_num_sp=self.max_sp, avg_sp_area=100)
            init_seed_list.append(init_seed.unsqueeze(0))

        s_init_seed = torch.cat(init_seed_list, 0)  # (shot, max_num_sp, 2)

        # OUTPUT
        depth = torch.unsqueeze(depth, 0)
        if (self.mode == 'train') | (self.mode == 'val'):
            return rgb, label, depth, s_x, s_y, s_init_seed, subcls_list, pos_hmap, cos_hmap, sin_hmap, widht_hmap,\
                   pos_hmap_grasp, cos_hmap_grasp, sin_hmap_grasp, widht_hmap_grasp, grasp_path, class_chosen
        else:
            return rgb, label, depth, s_x, s_y, s_init_seed, subcls_list, raw_label, pos_hmap, cos_hmap, sin_hmap, \
                   widht_hmap
