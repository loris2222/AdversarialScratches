from target import *
from optimizer import *
from attackmodel import *
from adversary import *
import utils

IMAGE_PATH = ""

def performattack():
    target = AzureCaptioning()
    attackmodel = BezierAttack(224, 500, 1, 3, True)
    optimizer = NGO(attackmodel.searchspace, 1, 10000)
    loss = ConfidenceLoss()

    adversary = Adversary(target, optimizer, loss, attackmodel)
    optimizer.setadversary(adversary)

    path = IMAGE_PATH
    pilim = Image.open(path)
    pilim = utils.torch_center_crop_resize(pilim, 224)
    im = np.array(pilim).reshape([1, 224, 224, 3])

    result, queries, params = adversary.performattack(im, maxqueries=1000)

    print(result)
    print(queries)
    print(params)


def lookforcaptions():
    for f in os.listdir("./azure/imagecaptions_ngo"):
        if "plane" not in f and "jet" not in f and "kite" not in f and "drone" not in f and "rocket" not in f:
            print(f)


def showoriginal():
    path = IMAGE_PATH
    pilim = Image.open(path)
    pilim = utils.torch_center_crop_resize(pilim, 224)
    im = Image.fromarray(np.array(pilim))
    im.show()

    target = AzureCaptioning()
    result = target.predict(np.array(pilim).reshape([1, 224, 224, 3]))
    print(result)
    print(result[0].captions[0].confidence)
    print(result[0].captions[0].text)


lookforcaptions()
