import numpy as np
import random
import os
from os import listdir
from os.path import isfile, join 

c_train  = list(np.load("train_classes_merged.npy"))
print(c_train)

np.save("all_classes.npy", np.array(list(range(88))))

c_val  = list(np.load("val_classes_merged.npy"))
print(c_val)

input_path = "/media/data/Datasets/graspnet/"
out_train = "train_big.txt"
out_val = "val_big.txt"


# CONTA SCENE
subfolders = [f.path for f in os.scandir(input_path) if f.is_dir()]
subfolders.sort()

file_t = open(out_train, "w")
file_v = open(out_val, "w")

tot = 0
for subfolder in subfolders:
    temp = subfolder.split("/")[-1]

    if (temp[0:6] != "scene_") | (len(temp) > 12):
        continue

    tot += 1
    if (int(temp[6:10]) < 190) & (int(temp[6:10]) >= 0):

        # print(int(temp[6:11]))
        print(temp[6:10])
        f_t = False
        f_v = False
        l = open(input_path + temp + "/object_id_list.txt", "r")
        for line in l:
            if int(line) in c_train:
            	f_t = True
            if int(line) in c_val:
            	f_v = True
        l.close()

        print("Train: ",f_t, " Val: ", f_v)

        
        for i in range(255):
        	if 20 > random.randint(0, 100):
        		if(int(temp[6:10]) < 130):
        			file_t.write(
	                    "scene_" + temp[6:10] + "/kinect/rgb/" + str(i).zfill(4) + ".png " +
	                    "scene_" + temp[6:10] + "/kinect/label/" +  str(i).zfill(4) + ".png " +
	                    "scene_" + temp[6:10] + "/kinect/depth/" +  str(i).zfill(4) + ".png " +
	                    "scene_" + temp[6:10] + "/kinect/rect/" +  str(i).zfill(4) + ".npy \n"
	                )
        		else:
        			file_v.write(
	                    "scene_" + temp[6:10] + "/kinect/rgb/" + str(i).zfill(4) + ".png " +
	                    "scene_" + temp[6:10] + "/kinect/label/" +  str(i).zfill(4) + ".png " +
	                    "scene_" + temp[6:10] + "/kinect/depth/" +  str(i).zfill(4) + ".png " +
	                    "scene_" + temp[6:10] + "/kinect/rect/" +  str(i).zfill(4) + ".npy \n"
	                )



        
file_t.close()
file_v.close()