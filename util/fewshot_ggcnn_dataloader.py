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
from skimage.filters import gaussian

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

        if self.mode == 'test':

            self.data_list = []

            list_read = open(self.data_root).readlines()

            for line in list_read:
                line = line.strip()
                line_split = line.split(' ')
                query_rgb = glob.glob(os.path.join(line_split[0],"rgb","*.png"))[0]
                query_depth = glob.glob(os.path.join(line_split[0],"depth","*.png"))[0]
                support_folder = line_split[1]
                supports_rgb = glob.glob(support_folder+"/rgb/*.png")
                supports_label = glob.glob(support_folder+"/label/*.png")
                target_obj = line_split[2]

                self.data_list.append([query_rgb, query_depth, supports_rgb, supports_label, target_obj])


        if (self.mode == 'train') | (self.mode == 'val'):

            # self.sub_list = list(np.load("/home/barcellona/workspace/git_repo/FSGGCNN/lists/new_lists/all_classes.npy"))
            # self.sub_val_list = list(np.load("/home/barcellona/workspace/git_repo/FSGGCNN/lists/new_lists/all_classes.npy"))
            self.sub_list = list(np.load("/home/bacchin/SemGraspNet/SemGraspNet_venv/FSGGCNN/lists/train_cls.npy"))
            self.sub_val_list = list(np.load("/home/bacchin/SemGraspNet/SemGraspNet_venv/FSGGCNN/lists/val_cls.npy"))
            print('sub_list: ', self.sub_list)
            print('sub_val_list: ', self.sub_val_list)

            # CREA DUE LISTE, UNA CONTENENTE LA COPPIA PATH IMMAGINE E PATH LABEL
            # L'ALTRA UNA UNA LISTA IN BASE AL NUMERO DI CLASSI CHE DEFINISCE QUALI IMMAGINI CONTENGONO QUELLA CLASSE 
            if self.mode == 'train':
                self.data_list, self.sub_class_file_list = make_dataset(data_root, data_list, self.sub_list)

                assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
            elif self.mode == 'val':
                self.data_list, self.sub_class_file_list = make_dataset(data_root, data_list, self.sub_val_list)
                assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list)

    def _get_crop_attrs(self, idx):
        gtbbs = GraspRectangles.load_from_graspnet_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 1280 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 720 - self.output_size))
        return center, left, top

    def get_gtbb(self, grasp_path, rot=0, zoom=1.0, obj_id=None, n_grasps=100, min_cdist=0, frict_th = 0.2):
        gtbbs = GraspRectangles.load_from_graspnet_file(grasp_path,
                                                        obj_id=obj_id, num_grasps = n_grasps, min_center_dist=min_cdist, friction = frict_th)  # num_grasps = grasps generated per object
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
        return gtbbs

    def __len__(self):

        return len(self.data_list)

    # RESITUTISCE UN IMMAGINE DI QUERY CON UN SUPPORTO CASUALE
    def __getitem__(self, index):

        if self.mode == 'test':
            query_rgb_path, query_depth_path, support_rgb_path_list, support_label_path_list, target_obj = self.data_list[index]

            rgb = cv2.imread(query_rgb_path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.float32(rgb)
            rgb = rgb[168:1105,513:1450,:]
            depth = cv2.imread(query_depth_path, cv2.IMREAD_GRAYSCALE)
            depth = np.float32(depth)
            depth = depth[168:1105,513:1450]

            class_chosen = int(target_obj)

            # CARICA IL VETTORE DI IMMAGINI E LABEL E LI SISTEMA (rende binari come prima)
            support_rgb_list = []
            support_label_list = []
            subcls_list = []
            for k in range(self.shot):

                subcls_list.append(class_chosen)
                support_rgb_path = support_rgb_path_list[k]
                support_label_path = support_label_path_list[k]
                support_rgb = cv2.imread(support_rgb_path, cv2.IMREAD_COLOR)
                support_rgb = cv2.cvtColor(support_rgb, cv2.COLOR_BGR2RGB)
                support_rgb = np.float32(support_rgb)
                support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
                target_pix = np.where(support_label == class_chosen)
                ignore_pix = np.where(support_label == 255)
                support_label[:, :] = 0
                support_label[target_pix[0], target_pix[1]] = 1
                support_label[ignore_pix[0], ignore_pix[1]] = 255
                if support_rgb.shape[0] != support_label.shape[0] or support_rgb.shape[1] != support_label.shape[1]:
                    raise (RuntimeError(
                        "Support Image & label shape mismatch: " + support_rgb_path + " " + support_label_path + "\n"))
                support_rgb_list.append(support_rgb)
                support_label_list.append(support_label)
            assert len(support_label_list) == self.shot and len(support_rgb_list) == self.shot

            rgb, _, depth, _= self.transform(rgb, depth)

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

            return rgb, depth, s_x, s_y, s_init_seed, subcls_list, class_chosen



        else:
            # LEGGE L'IMMAGINE E IL LABEL DI TARGET (target skippato per inference)
            label_class = []

            rgb_path, label_path, depth_path, grasp_path = self.data_list[index]  # 4 path to data

            # LEGGE LABEL, RGB, DEPTH, GRASP HEATMAP
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.float32(rgb)
            depth = cv2.imread(depth_path, -1)
            depth = np.float32(depth)

            # SCEGLIE IL LABEL DI TARGET
            # LEGGE LE CLASSI PRESENTI NELL`IMMAGINE
            label_class = np.unique(label).tolist()
            if 0 in label_class:
                label_class.remove(0)
            if 255 in label_class:
                label_class.remove(255)

            new_label_class = []
            for c in label_class:
                if c in self.sub_val_list:
                    if self.mode == 'val':
                        new_label_class.append(c)
                if c in self.sub_list:
                    if self.mode == 'train':
                        new_label_class.append(c)
                label_class = new_label_class
            assert len(label_class) > 0

            # SCEGLIE UNA CLASSE CASUALE DA FARE TRAINING
            class_chosen = label_class[random.randint(1, len(label_class)) - 1]
            class_chosen = class_chosen

            # DA PREPARARE LE 4 HEATMAP GT
            # Load the grasps of the selected objects
            bbs = self.get_gtbb(grasp_path, rot=0, zoom=1.0, obj_id=class_chosen, min_cdist=0, n_grasps = 300, frict_th=1.0)

            shape = (np.shape(label))
            pos_img_semantic, ang_img_semantic, width_img_semantic = bbs.draw(shape)
            width_img_semantic = np.clip(width_img_semantic, 0.0, self.gripper_width) / self.gripper_width

            #GUASSIAN BLUR
            #ang_img_semantic = gaussian(ang_img_semantic,10.0, preserve_range=True)
            
            pos_hmap = pos_img_semantic
            cos_hmap = np.cos(2 * ang_img_semantic)
            sin_hmap = np.sin(2 * ang_img_semantic)
            widht_hmap = width_img_semantic

            #normalize sin e cos
            cos_hmap = (cos_hmap - (-1)) / 2
            sin_hmap = (sin_hmap - (-1)) / 2

            # Load the grasps of all objects
            bbs = self.get_gtbb(grasp_path, rot=0, zoom=1.0, obj_id=None, min_cdist=0, n_grasps = 300, frict_th=1.0)
            shape = (np.shape(label))
            #print("**DATALOADER** bbs len: ", bbs.to_array().shape )
            pos_img_grasp, ang_img_grasp, width_img_grasp = bbs.draw(shape)
            width_img_grasp = np.clip(width_img_grasp, 0.0, self.gripper_width) / self.gripper_width

            #GUASSIAN BLUR
            #ang_img_grasp = gaussian(ang_img_grasp, 10.0, preserve_range=True)

            pos_hmap_grasp = pos_img_grasp
            cos_hmap_grasp = np.cos(2 * ang_img_grasp)
            sin_hmap_grasp = np.sin(2 * ang_img_grasp)
            widht_hmap_grasp = width_img_grasp

            #normalize sin e cos
            cos_hmap_grasp = (cos_hmap_grasp - (-1)) / 2
            sin_hmap_grasp = (sin_hmap_grasp - (-1)) / 2

            #WEIGHTED HEATMAP with ONE and all obejcts
            pos_hmap = pos_img_grasp*0.25 + pos_hmap*0.75

            # CREA LA MASCHERA BINARIA (0 pixel background 1 pixel classe 255 pixel per adattare dimensione)
            if (self.mode != 'inference'):
                target_pix = np.where(label == class_chosen)
                ignore_pix = np.where(label == 255)
                label[:, :] = 0
                if target_pix[0].shape[0] > 0:
                    label[target_pix[0], target_pix[1]] = 1
                label[ignore_pix[0], ignore_pix[1]] = 255

                # TROVA TUTTI I FILE DELLA CLASSE SCELTA
            file_class_chosen = self.sub_class_file_list[class_chosen]
            num_file = len(file_class_chosen)

            # SCEGLIE K IMMAGINI DI SUPPORTO CASUALI (k shot)
            support_rgb_path_list = []
            support_label_path_list = []
            support_idx_list = []
            for k in range(self.shot):
                support_idx = random.randint(1, num_file) - 1
                support_rgb_path = rgb_path
                support_label_path = label_path
                while ((
                               support_rgb_path == rgb_path and support_label_path == label_path) or support_idx in support_idx_list):
                    support_idx = random.randint(1, num_file) - 1
                    support_rgb_path, support_label_path, _, _ = file_class_chosen[support_idx]
                support_idx_list.append(support_idx)
                support_rgb_path_list.append(support_rgb_path)
                support_label_path_list.append(support_label_path)

            # CARICA IL VETTORE DI IMMAGINI E LABEL E LI SISTEMA (rende binari come prima)
            support_rgb_list = []
            support_label_list = []
            subcls_list = []
            for k in range(self.shot):
                if (self.mode == 'train'):
                    subcls_list.append(self.sub_list.index(class_chosen))
                else:
                    subcls_list.append(self.sub_val_list.index(class_chosen))
                    # print(self.sub_val_list.index(class_chosen))
                support_rgb_path = support_rgb_path_list[k]
                support_label_path = support_label_path_list[k]
                support_rgb = cv2.imread(support_rgb_path, cv2.IMREAD_COLOR)
                support_rgb = cv2.cvtColor(support_rgb, cv2.COLOR_BGR2RGB)
                support_rgb = np.float32(support_rgb)
                support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
                target_pix = np.where(support_label == class_chosen)
                ignore_pix = np.where(support_label == 255)
                support_label[:, :] = 0
                support_label[target_pix[0], target_pix[1]] = 1
                support_label[ignore_pix[0], ignore_pix[1]] = 255
                if support_rgb.shape[0] != support_label.shape[0] or support_rgb.shape[1] != support_label.shape[1]:
                    raise (RuntimeError(
                        "Support Image & label shape mismatch: " + support_rgb_path + " " + support_label_path + "\n"))
                support_rgb_list.append(support_rgb)
                support_label_list.append(support_label)
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
