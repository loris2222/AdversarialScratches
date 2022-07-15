import numpy as np
from PIL import Image
import utils

IMAGE_PATH = ""

path = IMAGE_PATH

im = Image.open(path)
im = utils.torch_center_crop_resize(im, 224)
im.show()

imarray = np.array(im)

# Linf and L0
# linf_perturb = np.random.normal(0, 10, [224, 224, 3])

# Patch L0
# linf_perturb = np.zeros([224, 224, 3])
# linf_perturb[60:100, 70:110] = np.random.normal(0, 255, [40, 40, 3])

# Any L0
# linf_perturb = np.zeros([224, 224, 3])
# x = np.random.uniform(0, 223, [200]).astype(int)
# y = np.random.uniform(0, 223, [200]).astype(int)
#
# for i in range(200):
#     linf_perturb[x[i], y[i], :] = np.random.uniform(0, 1, [1, 1, 3])*255

# Bezier
im = utils.superimpose_bezier(imarray, [(30, 30), (40, 80), (90, 90)], (0, 0, 1)).astype(np.uint8)
linf_perturb = (im - imarray)

imperturb = Image.fromarray((linf_perturb/2.0+127.5).astype(np.uint8))
imperturb.show()

linf_perturb = np.clip(linf_perturb + imarray.astype(int), 0, 255).astype(np.uint8)
imperturb = Image.fromarray(linf_perturb)
imperturb.show()

