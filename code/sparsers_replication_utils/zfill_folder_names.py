import os

basepath = os.getcwd()
target_path = os.path.join(basepath, "./imagenet/")
dataset_path = os.path.join(basepath, "./imagenet/raw/")
target_file_path = os.path.join(basepath, "./imagenet/ILSVRC2012_validation_ground_truth.txt")

for i in range(1, 1001):
    os.rename(os.path.join(target_path, "./" + str(i)), os.path.join(target_path, "./" + str(i).zfill(8)))