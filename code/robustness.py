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
from target import *
from attackmodel import *
from perturbation_sampler import *

EXP_NAME = ""

DATASET = "tsrd"

basepath = os.getcwd()
csvpath = os.path.join(basepath, "../experiments/" + EXP_NAME)

if DATASET == "imagenet":
    dataset_path = os.path.join(basepath, "../datasets/imagenet/val")
    labels_file_path = os.path.join(basepath, "../datasets/valnew.txt")
    IMSIZE = 224
    CLASSES = 1000
elif DATASET == "tsrd":
    dataset_path = os.path.join(basepath, "../datasets/tsrd/images")
    labels_file_path = os.path.join(basepath, "../datasets/tsrd/labels.txt")
    IMSIZE = 224
    CLASSES = 58
else:
    raise ValueError("invalid dataset name")

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

# ----------------------------------------------------------------------------------------------------------------------
target = ResNet50ScoreBased()  # ResNet50ScoreBased() PyTorchResNet50()
attackmodel = BoxedSquiggle(0, 20, 20)  # BezierAttack(500, 1, 1)  GeneralEll0BinaryColour(50)
# sigma = np.array(([3.0]*6 + [0]*3))  # np.array(([3.0]*6 + [0.1]*3)*3)  np.array(([3.0]*2 + [0.1]*3)*50)
perturbation = SquiggleRelativeGrid(attackmodel.searchspace, ([1, -1] + [0]*3 + ([0]*2)*20), 2, 8, 20)
# GaussianPerturbation(attackmodel.searchspace, sigma)  RigidMotion(attackmodel.searchspace, ([1, -1] + [0]*3)*50, 3.0)
# RigidMotion(attackmodel.searchspace, ([1, -1]*3 + [0]*3), 3.0)

PERTURB_ITER = 64
CHECK_ONLY = True

# ----------------------------------------------------------------------------------------------------------------------

total_fooled = 0
for line in reader:
    if targetidx is not None and line[targetidx] != line[origlabelidx]:
        skipped += 1
        continue
    if -1 != int(line[advlabelidx]):
        countrows += 1
        # Load the original image
        imname = str(line[nameidx])
        split = imname.split("_")
        split = split[-1].split(".")
        imnum = int(split[0])
        im = Image.open(os.path.join(dataset_path, imname)).convert('RGB')
        im = utils.torch_center_crop_resize(im)
        raw = np.array(im).astype(np.uint8).reshape([1, IMSIZE, IMSIZE, 3])
        # Double check that the image is correctly classified
        original_prediction = target.predict(raw).flatten()
        x = np.argmax(original_prediction)
        y = int(labels[imnum-1])
        if x != y:
            if original_prediction[x] == original_prediction[y]:
                print("Tie detected in original, skipping")
                countrows -= 1
                continue
            else:
                print("Skipping, labels should be the same")
                countrows -= 1
                continue

        # Attack the image and verify that model is fooled on the original perturbation
        params = line[paramidx:]
        params = [np.longdouble(par) for par in params]
        params = np.array(params).reshape([1, len(attackmodel.searchspace)])
        adv_image = attackmodel(raw, params).reshape(IMSIZE, IMSIZE, 3)
        adv_prediction = target.predict(adv_image).flatten()
        x = np.argmax(adv_prediction)
        if x == y:
            if adv_prediction[x] == adv_prediction[y]:
                print("Tie detected in adversarial, skipping")
                countrows -= 1
                continue
            else:
                print("Skipping, labels should be different")
                countrows -= 1
                continue

        # Run perturbation iterations and see whether it still holds
        fooled_iter = 0
        perturb_params = perturbation(params, PERTURB_ITER)
        for i in range(PERTURB_ITER):
            adv_image = attackmodel(raw, perturb_params[i:i+1, :]).reshape(IMSIZE, IMSIZE, 3)
            adv_prediction = target.predict(adv_image).flatten()
            x = np.argmax(adv_prediction)
            # if it still fools the model we can count it
            if x != y:
                fooled_iter += 1

            # Display the adversarial image
            if not CHECK_ONLY or i == 0 or i == PERTURB_ITER-1:
                Image.fromarray(adv_image).show()
                input("wait ")

        print("Sample "+str(imnum)+" was adversarial for "+str(fooled_iter)+"/"+str(PERTURB_ITER)+" tries")
        total_fooled += fooled_iter

print("On "+str(countrows)+" samples tested, a total of "+str(total_fooled)+" perturbations were successful.")
print("Robust adversarial rate: "+str(total_fooled/(countrows*PERTURB_ITER)))
