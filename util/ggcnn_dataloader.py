import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch
from torchvision.datasets.folder import IMG_EXTENSIONS

from tqdm import tqdm
import matplotlib.pyplot as plt
from util.GraspRectangle import GraspRectangles, GraspRectangle, Grasp


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(data_root=None, data_list=None, sub_list=None):
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    image_label_list = []
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')

        # DA AGGIUSTARE: ORDINE OGGETTI
        depth_name = os.path.join(data_root, line_split[2])
        grasp_name = os.path.join(data_root, line_split[3])  # path to rectangular grasps

        item = (depth_name, grasp_name)

        image_label_list.append(item)

    print("{} images&label pair after processing! ".format(len(image_label_list)))
    return image_label_list


def read_classes(class_path):
    classes = []
    file = open(class_path, "r")
    for line in file:
        classes.append(int(line))
    file.close()
    return classes


# data_root = "/media/data/Datasets/graspnet/scenes"
class GraspingData(Dataset):
    def __init__(self, shot=1, max_sp=5, data_root=None, data_list=None, data_classes=None, \
                 transform=None, transform_shots=None, mode='train', num_all_classes=None, gripper_width=150):

        assert mode in ['train', 'val']

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
            '''
            SERVE STA ROBA??? IO HO GIÀ LE 3 LISTE, NON CAPISCO... 
        
            self.class_list = self.graspnet_training_classes #[DA 1 A 28]
            self.sub_list, self.sub_val_list = create_split_list(self.graspnet_training_classes, self.split) #Crea la lista dividendo per 4

            print('sub_list: ', self.sub_list)
            print('sub_val_list: ', self.sub_val_list)        
            '''
            self.sub_list = list(np.load("/home/bacchin/SemGraspNet/SemGraspNet_venv/FSGGCNN/lists/train_cls.npy"))
            self.sub_val_list = list(np.load("/home/bacchin/SemGraspNet/SemGraspNet_venv/FSGGCNN/lists/val_cls.npy"))

            # CREA DUE LISTE, UNA CONTENENTE LA COPPIA PATH IMMAGINE E PATH LABEL
            # L'ALTRA UNA UNA LISTA IN BASE AL NUMERO DI CLASSI CHE DEFINISCE QUALI IMMAGINI CONTENGONO QUELLA CLASSE 
            if self.mode == 'train':
                self.data_list = make_dataset(data_root, data_list, self.sub_list)

            elif self.mode == 'val':
                self.data_list = make_dataset(data_root, data_list, self.sub_val_list)
                self.data_list = self.data_list[::2]

    def _get_crop_attrs(self, idx):
        gtbbs = GraspRectangles.load_from_graspnet_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 1280 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 720 - self.output_size))
        return center, left, top

    def get_gtbb(self, grasp_path, rot=0, zoom=1.0, obj_id=None):
        gtbbs = GraspRectangles.load_from_graspnet_file(grasp_path,
                                                        obj_id=obj_id, num_grasps = 100)  # da aggiungere la variabile giusta con il path
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
        return gtbbs

    def __len__(self):

        return len(self.data_list)

    # RESITUTISCE UN IMMAGINE DI QUERY CON UN SUPPORTO CASUALE
    def __getitem__(self, index):

        # LEGGE L'IMMAGINE E IL LABEL DI TARGET (target skippato per inference)
        label_class = []

        depth_path, grasp_path = self.data_list[index]  # 4 path to data

        depth = cv2.imread(depth_path, -1)
        depth = np.float32(depth)

        # DA PREPARARE LE 3 HEATMAP GT 
        # Load the grasps
        bbs = self.get_gtbb(grasp_path, rot=0, zoom=1.0, obj_id=None)
        shape = (np.shape(depth))
        pos_img, ang_img, width_img = bbs.draw(shape)
        width_img = np.clip(width_img, 0.0, self.gripper_width) / self.gripper_width

        pos_hmap = pos_img
        cos_hmap = np.cos(2 * ang_img)
        sin_hmap = np.sin(2 * ang_img)
        widht_hmap = width_img

        if self.transform is not None:
            depth, heatmaps = self.transform(depth, [pos_hmap, cos_hmap, sin_hmap,
                                                     widht_hmap])  # definiamo una classe transform2 per gestire x argomenti in input
            pos_hmap = heatmaps[0]
            cos_hmap = heatmaps[1]
            sin_hmap = heatmaps[2]
            widht_hmap = heatmaps[3]

        # OUTPUT
        depth = torch.unsqueeze(depth, 0)
        if (self.mode == 'train') | (self.mode == 'val'):
            return depth, pos_hmap, cos_hmap, sin_hmap, widht_hmap, grasp_path
        else:
            return depth, pos_hmap, cos_hmap, sin_hmap, widht_hmap
