import os

basepath = os.getcwd()
target_path = os.path.join(basepath, "./imagenet/")
dataset_path = os.path.join(basepath, "./imagenet/raw/")
target_file_path = os.path.join(basepath, "./imagenet/ILSVRC2012_validation_ground_truth.txt")

# for i in range(1, 1001):
#     os.mkdir(os.path.join(target_path, "./"+str(i)))

target_file = open(target_file_path, 'r')

lines = target_file.readlines()

for count in range(len(lines)):
    lines[count] = lines[count].strip("\n")

images = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

for count in range(1, 50001):
    os.rename(os.path.join(dataset_path, "./ILSVRC2012_val_"+str(count).zfill(8)+".JPEG"), os.path.join(target_path, "./" + str(lines[count-1]) + "/ILSVRC2012_val_"+str(count).zfill(8)+".JPEG"))
