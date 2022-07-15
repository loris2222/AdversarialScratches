from PIL import Image, ImageDraw
import numpy as np
import random
from attackmodel import BezierAttack
from defense import JPEGDefense, MedianFilterDefense

RESULT_PATH = ""
SAMPLE_PATH = ""
MASK_PATH = ""

def create_frame_img():
    path = RESULT_PATH

    img = Image.open(path)
    img = img.resize((224, 224), resample=0)

    draw = ImageDraw.Draw(img)

    for i in range(213,223):
        for j in range(0,223):
            draw.point((i,j),(np.random.randint(0,2)*255, np.random.randint(0,2)*255, np.random.randint(0,2)*255))

    for i in range(0,223):
        for j in range(213,223):
            draw.point((i,j),(np.random.randint(0,2)*255, np.random.randint(0,2)*255, np.random.randint(0,2)*255))

    for i in range(0,223):
        for j in range(0,10):
            draw.point((i,j),(np.random.randint(0,2)*255, np.random.randint(0,2)*255, np.random.randint(0,2)*255))

    for i in range(0,10):
        for j in range(0,223):
            draw.point((i,j),(np.random.randint(0,2)*255, np.random.randint(0,2)*255, np.random.randint(0,2)*255))

    img.show()


def create_perturb_img(type, img=None):
    # path = "N:/Documents/Uni/AssegnoRicerca/_ScratchThat/Images/ellnorms_cat/cat.JPEG"
    # path = "N:/Documents/Uni/AssegnoRicerca/_ScratchThat/Images/5origi.png"
    path = RESULT_PATH

    if img is None:
        img = Image.open(path)
        img = img.resize((224, 224), resample=0)

    draw = ImageDraw.Draw(img)

    if type == "none":
        pass

    if type == "l0":
        for i in range(200):
            x = np.random.randint(0, 223)
            y = np.random.randint(0, 223)

            draw.point((x,y), (np.random.randint(0, 2)*255, np.random.randint(0, 2)*255, np.random.randint(0, 2)*255))

    if type == "l2":
        for i in range(223):
            for j in range(223):
                color = img.getpixel((i, j))
                noise = (np.random.randint(0, 30), np.random.randint(0, 30), np.random.randint(0, 30))
                color = tuple(np.clip(np.array(color)+np.array(noise), 0, 255))

                draw.point((i, j), color)

    if type == "scratch":
        def gen_random_point(attackmodel):
            res = []
            for elem in attackmodel.searchspace:
                rand = random.uniform(elem[0], elem[1])
                res.append(rand)
            return res
        linecount = 1
        atkmod = BezierAttack(imsize=224, ell0=250, width=1, linecount=linecount, bynarize=True)
        param = [gen_random_point(atkmod)]
        batch = np.array(img).reshape([1, 224, 224, 3])
        img = atkmod(batch, param)
        img = Image.fromarray(img.reshape([224, 224, 3]))

    return img


def display_defended_image(img):
    jpeg85 = JPEGDefense(85)
    jpeg90 = JPEGDefense(90)
    jpeg95 = JPEGDefense(95)
    jpeg99 = JPEGDefense(99)
    median = MedianFilterDefense(3)

    defended = jpeg85.defend([img])
    defended[0].show()
    defended = jpeg90.defend([img])
    defended[0].show()
    defended = jpeg95.defend([img])
    defended[0].show()
    defended = jpeg99.defend([img])
    defended[0].show()
    defended = median.defend([img])
    defended[0].show()


def perturb_and_mask():
    sample_path = SAMPLE_PATH
    mask_path = MASK_PATH

    sample = Image.open(sample_path)
    sample = sample.resize((224, 224), resample=0)
    mask = Image.open(mask_path)
    mask = mask.resize((224, 224), resample=0)
    attacked = create_perturb_img("scratch", sample)

    masked_array = np.array(attacked)
    mask_array = np.array(mask)
    sample_array = np.array(sample)
    masked_array[mask_array == 0] = sample_array[mask_array == 0]

    masked = Image.fromarray(masked_array)

    sample.show()
    attacked.show()
    mask.show()
    masked.show()

# perturbed = create_perturb_img("none")
# display_defended_image(img)
# perturbed.show()

perturb_and_mask()

