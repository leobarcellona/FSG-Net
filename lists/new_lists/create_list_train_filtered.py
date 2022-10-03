import glob

import os
from os import listdir
from os.path import isfile, join
import numpy as np

input_path = "/media/data/Datasets/graspnet/"
input_classes = "/home/barcellona/workspace/git_repo/FSGGCNN/lists/new_lists/graspnet_classes.txt"
output_list = "/home/barcellona/workspace/git_repo/FSGGCNN/lists/new_lists/graspnet_train_filtered.txt"
output_list_val = "/home/barcellona/workspace/git_repo/FSGGCNN/lists/new_lists/graspnet_val_filtered.txt"
output_classes = "/home/barcellona/workspace/git_repo/FSGGCNN/lists/new_lists/graspnet_classes_filtered.txt"


# CREA LOS SPLIT
def create_split_list(class_list, split):
    elements = len(class_list)
    n_split = int(elements / 4)
    print("NUM elements: " + str(elements))
    print("NUM split: " + str(n_split))

    sub_val_list = []
    for i in range(n_split * split, n_split * (split + 1)):
        sub_val_list.append(class_list[i])

    sub_list = []
    for j in range(4):
        if j != split:
            for i in range(n_split * j, n_split * (j + 1)):
                sub_list.append(class_list[i])

    # sub_list = class_list - sub_val_list

    return sub_list, sub_val_list


objects = np.zeros(88)
objects_test = np.zeros(88)

# CONTA SCENE
subfolders = [f.path for f in os.scandir(input_path) if f.is_dir()]
subfolders.sort()

tot = 0
for subfolder in subfolders:
    temp = subfolder.split("/")[-1]

    if (temp[0:6] != "scene_") | (len(temp) > 12):
        continue

    tot += 1
    if (int(temp[6:10]) < 130) & (int(temp[6:10]) >= 0):

        # print(int(temp[6:11]))
        l = open(input_path + temp + "/object_id_list.txt", "r")
        for line in l:
            objects[int(line)] = 1
        l.close()

for subfolder in subfolders:
    temp = subfolder.split("/")[-1]

    if (temp[0:6] != "scene_") | (len(temp) > 12):
        continue

    tot += 1
    if (int(temp[6:10]) < 190) & (int(temp[6:10]) >= 160):

        # print(int(temp[6:11]))
        l = open(input_path + temp + "/object_id_list.txt", "r")
        for line in l:
            objects_test[int(line)] = 1
        l.close()

# Leggi classi usate ad asgnet
file_classes = open(input_classes, "r")
classes = []
for line in file_classes:
    classes.append(int(line))
file_classes.close()
print("classi: ", classes)

# togli lo split di train
split_t, split_v = create_split_list(classes, 3)

for o in split_t:
    objects[o] = 0

print(np.count_nonzero(objects))
"""
print(split_v)
print(np.count_nonzero(objects))
new = 0
for i in range(88):
    if (objects[i]==0) & (objects_test[i]==1):
        new += 1

print("Total: ", np.count_nonzero(objects_test))
print("New: ", new)
"""

# scene_0031/kinect/rgb/0000.png scene_0031/kinect/label/0000.png scene_0031/kinect/depth/0000.png scene_0031/kinect/rect/0000.npy

n = 0
file = open(output_list, "w")
file_val = open(output_list_val, "w")
frame = 0
for subfolder in subfolders:

    subfolder = subfolder.split("/")[-1]
    if (subfolder[0:6] != "scene_") | (len(subfolder) > 10):
        continue

    if n > 130:
        break

    frame += 1

    files_rgb = [f for f in listdir(input_path + subfolder + "/kinect/rgb") if
                 isfile(join(input_path + subfolder + "/kinect/rgb", f))]
    files_rgb.sort()
    files_seg = [f for f in listdir(input_path + subfolder + "/kinect/label") if
                 isfile(join(input_path + subfolder + "/kinect/label", f))]
    files_seg.sort()

    if frame % 5 != 0:
        for i in range(len(files_rgb)):
            if i % 8 == 0:
                file.write(
                    subfolder + "/kinect/rgb/" + files_rgb[i].split("/")[-1] + " " +
                    subfolder + "/kinect/label/" + files_seg[i].split("/")[-1] + " " +
                    subfolder + "/kinect/depth/" + files_seg[i].split("/")[-1].split(".")[0] + ".png " +
                    subfolder + "/kinect/rect/" + files_seg[i].split("/")[-1].split(".")[0] + ".npy \n"
                )
    else:
        for i in range(len(files_rgb)):
            if i % 5 == 0:
                file_val.write(
                    subfolder + "/kinect/rgb/" + files_rgb[i].split("/")[-1] + " " +
                    subfolder + "/kinect/label/" + files_seg[i].split("/")[-1] + " " +
                    subfolder + "/kinect/depth/" + files_seg[i].split("/")[-1].split(".")[0] + ".png " +
                    subfolder + "/kinect/rect/" + files_seg[i].split("/")[-1].split(".")[0] + ".npy \n"
                )
    n += 1

file.close()
file_val.close()

file_classes = open(output_classes, "w")
for i in range(objects.shape[0]):
    if objects[i] == 1:
        file_classes.write(str(i) + "\n")

file_classes.close()
