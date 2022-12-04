import os
import shutil

dataset_loc = "../datasets/Sohas_weapon-Detection/images_test/"
dataset_new_loc = "../datasets/Sohas_weapon-Detection/split_images_test/"

files = os.listdir(dataset_loc)
current_dir = "0"
print(len(files))

for file_name in files:
    shutil.copy(dataset_loc+file_name, dataset_new_loc+current_dir+"/"+file_name)
    if current_dir == "0":
        current_dir = "1"
    else:
        current_dir = "0"

files = os.listdir(dataset_new_loc+"0")
print("# of files in '0':", len(files))
files = os.listdir(dataset_new_loc+"1")
print("# of files in '1':", len(files))