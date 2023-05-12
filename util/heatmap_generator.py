import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
from os import listdir
from os.path import isfile, join, isdir
from multiprocessing import Pool

import sys

from tqdm import tqdm

sys.path.append('../FSGGCNN/util')
print(sys.path)

from GraspRectangle import GraspRectangles, GraspRectangle, Grasp


def get_gtbb(grasp_path, rot=0, zoom=1.0, obj_id=None, n_grasps=100, min_cdist=0, frict_th=0.2):
    gtbbs = GraspRectangles.load_from_graspnet_file(grasp_path,
                                                    obj_id=obj_id, num_grasps=n_grasps, min_center_dist=min_cdist,
                                                    friction=frict_th)  # num_grasps = grasps generated per object
    return gtbbs

def generate_hmap(dir, path_base):

    if dir[:6] != "scene_":
        return
    print(dir)
    path = path_base + dir + "/kinect/"

    out_path = path + "heatmpas/"
    try:
        os.mkdir(out_path)
    except OSError:
        print("It exist")

    path_ann = path + "rect/"
    frames = [f for f in listdir(path_ann) if isfile(join(path_ann, f))]
    frames.sort()
    for f in tqdm(frames):
        # print(f)

        name = f.split(".")[0]

        grasp_path = path_ann + f
        bbs = get_gtbb(grasp_path, rot=0, zoom=1.0, obj_id=None, min_cdist=0, n_grasps=300, frict_th=1.0)
        shape = (720, 1280)
        pos_img_grasp, ang_img_grasp, width_img_grasp = bbs.draw(shape)

        width_img_grasp = np.clip(width_img_grasp, 0.0, 150) / 150
        width_hmap_grasp = np.floor(width_img_grasp * 15)

        angle_hmap_grasp = (ang_img_grasp + math.pi / 2) / math.pi
        angle_hmap_grasp = np.floor(angle_hmap_grasp * 18)
        angle_hmap_grasp[angle_hmap_grasp == 18] = 17

        cv2.imwrite(out_path + name + "_pos.png", pos_img_grasp)
        cv2.imwrite(out_path + name + "_ang.png", angle_hmap_grasp)
        cv2.imwrite(out_path + name + "_wid.png", width_hmap_grasp)


def run():
    path_base = "/media/data/Datasets/graspnet/scenes/"
    dirs = [f for f in listdir(path_base) if isdir(join(path_base, f))]
    dirs.sort()
    pool = Pool(8)
    for dir in dirs:
        pool.apply_async(func = generate_hmap, args = (dir, path_base))
    pool.close()
    pool.join()


if __name__ == '__main__':
    run()
