import os

basepath = os.getcwd()
target_path = os.path.join(basepath, "./imagenet/")
dataset_path = os.path.join(basepath, "./imagenet/raw/")
target_file_path = os.path.join(basepath, "./imagenet/ILSVRC2012_validation_ground_truth.txt")

for i in range(1, 1001):
    images = [f for f in os.listdir(os.path.join(target_path, "./" + str(i)))]
    for image in images:
        os.rename(os.path.join(target_path, "./" + str(i) + "/" + image),
                  os.path.join(dataset_path, "./" + image))



