# Takes the first 500 images in the imagenet validation set and puts them in classname-folders to be used elsewhere
import os
from shutil import copyfile

SAMPLES = 100

IMAGENET_PATH = ""
LABELS_PATH = ""

TARGET_PATH = ""

labels_file = open(LABELS_PATH, 'r')
labels = labels_file.readlines()

images = sorted([f for f in os.listdir(IMAGENET_PATH) if os.path.isfile(os.path.join(IMAGENET_PATH, f))])

# Creates 1000 folders in target folder
for i in range(1000):  # range(1, 1001)
    os.mkdir(os.path.join(TARGET_PATH, str(i).zfill(8)))

for i in range(SAMPLES):
    copyfile(os.path.join(IMAGENET_PATH, images[i]), os.path.join(TARGET_PATH, labels[i].replace('\n','').zfill(8) + "/" + images[i]))  # str(int(labels[i].replace('\n',''))+1)
