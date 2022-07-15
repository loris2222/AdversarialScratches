import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from PIL import Image
import utils
from optimizer import *
from adversary import *
from defense import *
from target import *
from attackmodel import *

# ----------------------------------------------------------------------------------------------------------------------

DATASET = "imagenet"
SAMPLES = 1000

IMAGENET_PATH = ""
IMAGENET_LABELS_FILE_PATH = ""

basepath = os.getcwd()
csvpath = os.path.join(basepath, "../experiments/search_strategy/ngo0.csv")

if DATASET == "imagenet":
    dataset_path = IMAGENET_PATH
    labels_file_path = IMAGENET_LABELS_FILE_PATH
    IMSIZE = 224
    IMCHANNELS = 3
    CLASS_COUNT = 1000
else:
    raise ValueError("Invalid dataset name")

target = PyTorchResNet50()  # ResNet50ScoreBased()  PyTorchResNet50()  PyTorchGTSRB()  PyTorchTSRD()
defense = JPEGDefense(85)  # MedianFilterDefense(3) JPEGDefense(85)  NoDefense()

# ----------------------------------------------------------------------------------------------------------------------

images = sorted([f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))])

labels_file = open(labels_file_path, 'r')
labels = labels_file.readlines()

clean_test_set = np.zeros([SAMPLES, IMSIZE, IMSIZE, 3]).astype(np.uint8)
defended_test_set = np.zeros([SAMPLES, IMSIZE, IMSIZE, 3]).astype(np.uint8)
defended_preds = np.zeros([SAMPLES, CLASS_COUNT])
clean_preds = np.zeros([SAMPLES, CLASS_COUNT])
for i in range(SAMPLES):
    # Load RGB image
    im = Image.open(os.path.join(dataset_path, images[i])).convert('RGB')
    im = utils.torch_center_crop_resize(im, IMSIZE)
    defim = defense.defend([im])[0]
    raw = np.array(im).astype(np.uint8)
    defended = np.array(defim).astype(np.uint8)
    clean_test_set[i, :, :, :] = raw
    defended_test_set[i, :, :, :] = defended

for i in range(SAMPLES):
    clean_preds[i, :] = target.predict(clean_test_set[i:i + 1, :, :, :])
    defended_preds[i, :] = target.predict(defended_test_set[i:i + 1, :, :, :])

clean_correct = 0
defended_correct = 0
for count in range(SAMPLES):
    clean_x = np.argmax(clean_preds[count, :])
    defended_x = np.argmax(defended_preds[count, :])
    y = int(labels[count])

    if clean_x == y:
        clean_correct += 1
    if defended_x == y:
        defended_correct += 1

print("Clean performance: " + str(clean_correct/SAMPLES))
print("Defended performance: " + str(defended_correct/SAMPLES))