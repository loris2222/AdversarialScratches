import os

basepath = os.getcwd()
target_path = os.path.join(basepath, "./imagenet/")
dataset_path = os.path.join(basepath, "./imagenet/raw/")
target_file_path = os.path.join(basepath, "./imagenet/val.txt")
new_file = open(os.path.join(basepath, "./imagenet/valnew.txt"), 'w')

# for i in range(1, 1001):
#     os.mkdir(os.path.join(target_path, "./"+str(i)))

target_file = open(target_file_path, 'r')

lines = target_file.readlines()

for line in lines:
    split = line.split(" ")
    line = split[1].strip("\n")
    new_file.writelines([line])

new_file.close()
