import tensorflow as tf
import keras
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
import os
from PIL import Image
import utils
import random

# ----------------------------------------------------------------------------------------------------------------------
#
# This is a proof of concept for the bezier modification of an image for a resnet50 model. It is in no way part of the
# framework.
#
# ----------------------------------------------------------------------------------------------------------------------


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMAGE_SIZE = (224, 224)
BEZIER_DEGREE = 2
MAX_ITERATIONS = 1000

def random_xys():
    xys = [(random.randrange(0, IMAGE_SIZE[0]), random.randrange(0, IMAGE_SIZE[1])) for i in range(BEZIER_DEGREE+1)]
    return xys


basepath = os.getcwd()
image_path = os.path.join(basepath, "../images/5.jpeg")
classname_path = os.path.join(basepath, "../imagenet_classes.txt")

classnames = open(classname_path).readlines()

im = Image.open(image_path)
im = im.resize((224, 224))
np_im = np.array(im)

model = ResNet50(weights='imagenet')

original_class = np.argmax(model(np_im.reshape([1, 224, 224, 3])))
print("Original class:")
print(classnames[original_class])
adv_class = original_class

count = 0
print("Attacking, iteration: ")
while adv_class == original_class:
    adv_im_np = utils.superimpose_bezier(np_im, random_xys(), (255, 255, 255))
    adv_class = np.argmax(model(adv_im_np.reshape([1, 224, 224, 3])))
    count += 1
    if count % 100 == 1:
        print("\r" + str(count-1), end="")
    if count > MAX_ITERATIONS:
        break

print("\n")
print("Adversarial class after "+str(count)+" iterations:")
print(classnames[adv_class])
adv_im = Image.fromarray(adv_im_np)
adv_im.show()
