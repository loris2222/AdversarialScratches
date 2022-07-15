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
IMAGENET_PATH = ""
IMAGENET_LABELS_FILE_PATH = ""

basepath = os.getcwd()
csvpath = os.path.join(basepath, "../experiments/search_strategy/ngo0.csv")

if DATASET == "imagenet":
    dataset_path = IMAGENET_PATH
    labels_file_path = IMAGENET_LABELS_FILE_PATH
    IMSIZE = 224
    IMCHANNELS = 3
else:
    raise ValueError("Invalid dataset name")

target = PyTorchResNet50()  # ResNet50ScoreBased()  PyTorchResNet50()  PyTorchGTSRB()  PyTorchTSRD()
attackmodel = BezierAttack(IMSIZE, 133, 1, 3, True)
defense = MedianFilterDefense(3)  # JPEGDefense(85) MedianFilterDefense(3)

# ----------------------------------------------------------------------------------------------------------------------

labels_file = open(labels_file_path, 'r')
labels = labels_file.readlines()

csvfile = open(csvpath, 'r', newline='')
reader = csv.reader(csvfile, delimiter=',')

countdiffer = 0
iter = []
countrows = 0
ell0 = []
skipped = 0

# skip header
header = next(reader)
queriesidx = header.index("queries")
try:
    ell0idx = header.index("ell0")
except ValueError:
    ell0idx = None
try:
    targetidx = header.index("target label")
except ValueError:
    targetidx = None
origlabelidx = header.index("original_label")
advlabelidx = header.index("adversarial_label")
nameidx = header.index("image name")
paramidx = header.index("0..n params")

diffsum = 0
totalsamples = 0
defended_count = 0

for line in reader:
    if targetidx is not None and line[targetidx] != line[origlabelidx]:
        skipped += 1
        continue
    countrows += 1
    if -1 != int(line[advlabelidx]):
        # Load the original image
        imname = str(line[nameidx])
        split = imname.split("_")
        split = split[-1].split(".")
        imnum = int(split[0])
        y = int(labels[imnum-1])

        # Load RGB Image
        raw = np.zeros([1, IMSIZE, IMSIZE, IMCHANNELS]).astype(np.uint8)
        im = Image.open(os.path.join(dataset_path, imname)).convert('RGB')
        # TODO is this torch_center_crop_resize?? (I think so)
        im = utils.torch_center_crop_resize(im, IMSIZE)
        raw[:, :, :, 0:3] = np.array(im).astype(np.uint8).reshape([1, IMSIZE, IMSIZE, 3])

        # Double check that the image is correctly classified
        original_prediction = target.predict(raw[:, :, :, 0:3]).flatten()
        x = np.argmax(original_prediction)
        if x != y:
            if original_prediction[x] == original_prediction[y]:
                print("Tie detected in original, continuing")
                continue
            else:
                print(original_prediction[x])
                print(original_prediction[y])
                print(imname)
                print("Original p "+str(x) + " Expected "+str(y))
                print("Error, labels should be the same")
                continue

        # Attack the image and verify that model is fooled
        params = line[paramidx:]
        params = [float(par) for par in params]
        params = np.array(params).reshape([1, len(attackmodel.searchspace)])
        adv_image = attackmodel(raw, params).reshape(1, IMSIZE, IMSIZE, 3)
        adv_prediction = target.predict(adv_image).flatten()
        x = np.argmax(adv_prediction)
        if x == y:
            print(imname)
            print("Original p " + str(x) + " Output " + str(y))
            print("Error, labels should be different")
            continue

        # Defend the image and see whether it is still mispredicted
        defim = defense.defend([Image.fromarray(adv_image.reshape([IMSIZE, IMSIZE, 3]))])[0]
        defended = np.array(defim).astype(np.uint8).reshape([1, IMSIZE, IMSIZE, 3])

        def_prediction = target.predict(defended).flatten()
        x = np.argmax(def_prediction)
        if x == y:
            defended_count += 1

        diff = utils.pixel_ell0_norm(adv_image.reshape([IMSIZE, IMSIZE, 3]), np.array(im))
        diffsum += diff
        totalsamples += 1


print("Average ell0: " + str(diffsum/totalsamples))
print("Recovery rate: " + str(defended_count/totalsamples))