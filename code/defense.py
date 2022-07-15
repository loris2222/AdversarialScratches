from abc import ABC, abstractmethod
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from io import BytesIO


class Defense(ABC):
    def __init__(self):
        pass

    @abstractmethod
    # Receives a list of PIL images and returns their modified versions according to the defense strategy
    def defend(self, images):
        pass


class JPEGDefense(Defense):
    # JPEG quality between 0 and 100
    def __init__(self, quality: int):
        super().__init__()
        self.quality = quality

    def defend(self, images):
        res = []
        for image in images:
            out = BytesIO()
            image.save(out, format="JPEG", quality=self.quality)
            out.seek(0)
            res.append(Image.open(out))

        return res


class MedianFilterDefense(Defense):
    # JPEG quality between 0 and 100
    def __init__(self, filtersize: int):
        super().__init__()
        self.filtersize = filtersize

    def defend(self, images):
        res = []
        for image in images:
            out = image.filter(ImageFilter.MedianFilter(size=self.filtersize))
            res.append(out)

        return res


class NoDefense(Defense):
    def __init__(self):
        super().__init__()

    def defend(self, images):
        return images


# Gives a sample of defences
if __name__ == "__main__":
    import os
    from attackmodel import *
    import utils
    sample = os.path.join(os.getcwd(), "../datasets/imagenet/val/ILSVRC2012_val_00000153.JPEG")
    im = Image.open(sample)

    im = np.array(utils.torch_center_crop_resize(im, 224)).astype(np.uint8).reshape([1, 224, 224, 3])
    params = [[np.random.randint(0, 224) for _ in range(6)] + [np.random.randint(0, 2) for _ in range(3)] for _ in range(3)]
    params = np.array(params).flatten().reshape([1, 27])

    attack = BezierAttack(224, 130, 1, 3, True)

    # Show original
    Image.fromarray(im.reshape([224, 224, 3])).show()

    # Attack and print
    imperturbed = attack(im, params)
    imperturbed = Image.fromarray(imperturbed.reshape([224, 224, 3]))
    imperturbed.show()

    # Median defense
    defense = MedianFilterDefense(3)
    defendedim = defense.defend([imperturbed])[0]
    defendedim.show()

    # JPEG defense
    defense = JPEGDefense(85)
    defendedim = defense.defend([imperturbed])[0]
    defendedim.show()

