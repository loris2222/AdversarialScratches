import csv
import os
from re import I
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from PIL import Image
import utils
from optimizer import *
from adversary import *
from target import *
from attackmodel import *
import argparse

# ----------------------------------------------------------------------------------------------------------------------

# parser = argparse.ArgumentParser(description='Performs attacks on the imagenet validation dataset and saves result')
# parser.add_argument("file")
# args = parser.parse_args()

EXP_NAME = ""
IMAGENET_PATH = ""
IMAGENET_LABELS_FILE_PATH = ""
GTSRD_PATH = ""

CHECK_ONLY = True
DATASET = "imagenet"  # "imagenet"  "masked_tsrd"

basepath = os.getcwd()
csvpath = os.path.join(basepath, "../experiments/scratch_count/" + EXP_NAME)  # args.file  # os.path.join(basepath, "../experiments/ex0.csv")

if DATASET == "imagenet":
    dataset_path = IMAGENET_PATH
    labels_file_path = IMAGENET_LABELS_FILE_PATH
    IMSIZE = 224
    IMCHANNELS = 3
elif DATASET == "gtsrb":
    dataset_path = os.path.join(basepath, "../datasets/gtsrb/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images")
    labels_file_path = os.path.join(basepath, "../datasets/gtsrb/test_labels.txt")
    IMSIZE = 32
    IMCHANNELS = 3
elif DATASET == "tsrd":
    dataset_path = os.path.join(basepath, "../datasets/tsrd/images")
    labels_file_path = os.path.join(basepath, "../datasets/tsrd/labels.txt")
    IMSIZE = 224
    IMCHANNELS = 3
elif DATASET == "masked_tsrd":
    dataset_path = os.path.join(basepath, "../datasets/tsrd_mask/test")
    mask_path = os.path.join(basepath, "../datasets/tsrd_mask/testmask")
    labels_file_path = os.path.join(basepath, "../datasets/tsrd_mask/labels.txt")
    IMSIZE = 224
    IMCHANNELS = 4
else:
    raise ValueError("Invalid dataset name")

target = PyTorchResNet50()  # ResNet50ScoreBased()  PyTorchResNet50()  PyTorchGTSRB()  PyTorchTSRD()
attackmodel = BezierAttack(IMSIZE, 80, 1, 5, True)  #  NonParametricPatchAttack(IMSIZE, 20)  PatchAttack(IMSIZE, 20, 20, True) MaskedPatchAttack(IMSIZE, 20, 20, True) MaskedBezierAttack(IMSIZE, 130, 1, 3, True) # BezierAttack(IMSIZE, 500, 1, 3)  # BezierAttack(IMSIZE, 50, 1, 1)  # BezierAttack(IMSIZE, 500, 1, 3)  GeneralEll0BinaryColour(IMSIZE, 50)

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

area = 0

for line in reader:
    if targetidx is not None and line[targetidx] != line[origlabelidx]:
        skipped += 1
        continue
    countrows += 1
    if -1 != int(line[advlabelidx]):
        # Load the original image
        imname = str(line[nameidx])
        if DATASET == "imagenet":
            split = imname.split("_")
            split = split[-1].split(".")
            imnum = int(split[0])
            y = int(labels[imnum-1])
        elif DATASET == "gtsrb":
            split = imname.split(".")
            imnum = int(split[0])
            y = int(labels[imnum])
        elif DATASET == "tsrd" or DATASET == "masked_tsrd":
            split = imname.split("_")
            y = int(split[0])
        else:
            raise ValueError("invalid dataset name")

        # Load RGB Image
        raw = np.zeros([1, IMSIZE, IMSIZE, IMCHANNELS]).astype(np.uint8)
        im = Image.open(os.path.join(dataset_path, imname)).convert('RGB')
        # TODO is this torch_center_crop_resize??
        im = utils.torch_center_crop_resize(im, IMSIZE)
        raw[:, :, :, 0:3] = np.array(im).astype(np.uint8).reshape([1, IMSIZE, IMSIZE, 3])

        # Load also mask for the other case
        if DATASET == "masked_tsrd":
            mask = Image.open(os.path.join(mask_path, imname))
            mask = utils.center_crop_resize(mask, IMSIZE)
            raw[:, :, :, 3] = np.array(mask).astype(np.uint8)

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

        diff = utils.pixel_ell0_norm(adv_image.reshape([IMSIZE, IMSIZE, 3]), np.array(im))
        diffsum += diff
        area += utils.perturb_area(raw[:, :, :, 0:3], adv_image)

        totalsamples += 1

        # Display the adversarial image
        if not CHECK_ONLY:
            # imarray = raw[0, :, :, 0:3] * np.repeat(raw[0, :, :, 3].reshape([IMSIZE, IMSIZE, 1]), 3, axis=2)
            # imarray = (adv_image.reshape([IMSIZE, IMSIZE, 3]) - raw[0, :, :, 0:3]) * ((np.repeat(raw[0, :, :, 3].reshape([IMSIZE, IMSIZE, 1]), 3, axis=2) - 1)*(-1)).astype(np.uint8)
            # Image.fromarray(imarray).show()
            # im.show()
            Image.fromarray(adv_image.reshape([IMSIZE, IMSIZE, 3])).show()
            # input(str(diff) + " " + str(x) + " " + str(y) + " - wait ")

print("Average ell0: " + str(diffsum/totalsamples))
print("Average area: " + str(area/totalsamples))