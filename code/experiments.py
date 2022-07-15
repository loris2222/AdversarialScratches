from optimizer import *
from adversary import *
from target import *
from attackmodel import *
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from PIL import Image
import csv
import argparse
import utils
import tensorflow as tf
import wandb

DEBUG = False

# This is set automatically for some dataset options and defines whether loaded images have to be converted to
# normalized tensors or if they can be used as numpy arrays.
USE_TORCH_IMAGES = False

# config = tf.config.experimental.per_process_gpu_memory_fraction = 0.1

# ARG PARSE
parser = argparse.ArgumentParser(description='Performs attacks on the imagenet validation dataset and saves result')
parser.add_argument("dataset")
parser.add_argument("samples")
parser.add_argument("batchsize")
parser.add_argument("maxiter")
parser.add_argument("model")
# parser.add_argument("target")
# parser.add_argument("loss")
parser.add_argument("attack")
parser.add_argument("atkparams")
parser.add_argument("optimizer")
parser.add_argument("optparams")
parser.add_argument("output")
parser.add_argument("wandb")
parser.add_argument("seed")
args = parser.parse_args()


# Fix seed for reproducibility
print("Setting random seed to "+ args.seed)
random.seed(int(args.seed))
np.random.seed(int(args.seed))

SAMPLES = int(args.samples)
BATCH_SIZE = int(args.batchsize)
MAX_ITER = int(args.maxiter)
ATK_PARAMS = args.atkparams
OPT_PARAMS = args.optparams
basepath = os.getcwd()

if str(args.wandb).lower() == "true":
    WANDB = True
else:
    WANDB = False

infostring = "Wandb framework is " + ("active" if WANDB else "inactive")

if str(args.dataset).lower() == "imagenet":
    DATASET = "imagenet"
    IMSIZE = 224
    IMCHANNELS = 3
    CLASS_COUNT = 1000
elif str(args.dataset).lower() == "gtsrb":
    DATASET = "gtsrb"
    IMSIZE = 32
    IMCHANNELS = 3
    CLASS_COUNT = 43
elif str(args.dataset).lower() == "tsrd":
    DATASET = "tsrd"
    IMSIZE = 224
    IMCHANNELS = 3
    CLASS_COUNT = 58
elif str(args.dataset).lower() == "masked_tsrd":
    DATASET = "masked_tsrd"
    IMSIZE = 224
    IMCHANNELS = 4
    CLASS_COUNT = 58
else:
    raise ValueError("Invalid dataset name, can either be 'imagenet' or 'gtsrb'")

infostring += "\nRunning on " + str(SAMPLES) + " samples from " + DATASET + " dataset"

infostring += "\nBatch size set to " + str(BATCH_SIZE)
infostring += "\nMaximum iterations set to " + str(MAX_ITER)

# Framework setup ------------------------------------------------------------------------------------------------------

if WANDB:
    wandb.init(project='scratchthat', entity='loris2222', config={
        "samples": SAMPLES,
        "batch size": BATCH_SIZE,
        "max iterations": MAX_ITER,
        "attack": args.attack,
        "attack params": args.atkparams,
        "optimizer": args.optimizer,
        "optimizer params": args.optparams,
        "output file": args.output,
    })
    config = wandb.config
    run_name = wandb.run.name
else:
    run_name = "not-synced"

# Fixes output file if set to auto
if args.output == "auto":
    args.output = "../experiments/"+args.samples+"samples_"+args.maxiter+"iter_"+args.model+"model_"\
                  + args.attack+args.atkparams+"_"+args.optimizer+args.optparams+"_"+run_name+".csv"
elif args.output == "test":
    args.output = "../experiments/ex0.csv"

infostring += "\nOutputting to " + os.path.join(basepath, args.output)

# Target model to be attacked (models are available in target.py)
if args.model == "torch":
    target = PyTorchResNet50()
elif args.model == "torch_n":
    if args.attack != "bezier_n":
        raise ValueError("Cannot use this attack with a normalized model, it is not supported")
    USE_TORCH_IMAGES = True
    target = PyTorchResNet50Normalized()
elif args.model == "keras":
    raise ValueError("keras models are no longer supported, but fixes should not take too long")
    # target = ResNet50ScoreBased()
elif args.model == "gtsrb":
    target = PyTorchGTSRB()
elif args.model == "tsrd":
    target = PyTorchTSRD()
else:
    print("Invalid target model '" + args.model + "'")
    exit(1)

# Loss function to use for the optimizer (available in adversary.py)
if USE_TORCH_IMAGES:
    loss = TorchMarginLoss()
else:
    loss = MarginLoss()  # SparseRSLoss(CLASS_COUNT) MarginLoss()  # UntargetedScoreLoss()

# Attack model used to translate search space points into adversarial images (in attackmodel.py)
ell0=-1
if args.attack == "bezier":
    split = ATK_PARAMS.split('-')
    ell0 = int(split[0])
    width = int(split[1])
    count = int(split[2])
    try:
        degree = int(split[3])
    except IndexError:
        degree = 2
    infostring += "\nUsing bezier attack with " + str(count) + " lines, ell0 bound " + str(ell0) + " width " + str(width) + " and degree " + str(degree)
    attackmodel = BezierAttack(IMSIZE, int(ell0/width), width, count, True, degree)
elif args.attack == "onecolorbezier":
    split = ATK_PARAMS.split('-')
    ell0 = int(split[0])
    width = int(split[1])
    count = int(split[2])
    infostring += "\nUsing one color bezier attack with " + str(count) + " lines, ell0 bound " + str(ell0) + " and width " + str(width)
    attackmodel = OneColorBezierAttack(IMSIZE, int(ell0/width), width, count, True)
elif args.attack == "imagecolorbezier":
    split = ATK_PARAMS.split('-')
    ell0 = int(split[0])
    width = int(split[1])
    count = int(split[2])
    infostring += "\nUsing image color bezier attack with " + str(count) + " lines, ell0 bound " + str(ell0) + " and width " + str(width)
    attackmodel = ImageColorBezierAttack(IMSIZE, int(ell0/width), width, count)
elif args.attack == "grayscalebezier":
    split = ATK_PARAMS.split('-')
    ell0 = int(split[0])
    width = int(split[1])
    count = int(split[2])
    infostring += "\nUsing grayscale bezier attack with " + str(count) + " lines, ell0 bound " + str(ell0) + " and width " + str(width)
    attackmodel = GrayscaleBezierAttack(IMSIZE, int(ell0/width), width, count, False)
elif args.attack == "spline":
    split = ATK_PARAMS.split('-')
    ell0 = int(split[0])
    width = int(split[1])
    count = int(split[2])
    degree = int(split[3])
    infostring += "\nUsing one color bezier attack with " + str(count) + " lines, ell0 bound " + str(ell0) + " width " + str(width) + " and degree " + str(degree)
    attackmodel = SplineAttack(IMSIZE, degree, int(ell0/width), width, count, True)
elif args.attack == "mcbezier":
    split = ATK_PARAMS.split('-')
    ell0 = int(split[0])
    width = int(split[1])
    infostring += "\nUsing MULTICOLOR bezier attack with ell0 bound " + str(ell0) + " and width " + str(width)
    attackmodel = MulticolorBezierAttack(IMSIZE, int(ell0 / width), width)
elif args.attack == "bcbezier":
    split = ATK_PARAMS.split('-')
    ell0 = int(split[0])
    width = int(split[1])
    infostring += "\nUsing BINARY COLOR bezier attack with ell0 bound " + str(ell0) + " and width " + str(width)
    attackmodel = BinaryColorBezierAttack(IMSIZE, int(ell0 / width), width)
elif args.attack == "flline":
    ell0 = int(ATK_PARAMS)
    infostring += "\nUsing fixed length straight line attack with length " + str(ell0)
    attackmodel = FixedLengthStraightLine(IMSIZE, ell0)
elif args.attack == "clline":
    ell0 = int(ATK_PARAMS)
    infostring += "\nUsing bounded straight line attack with ell0 bound " + str(ell0)
    attackmodel = ConstrainedLengthStraightLine(IMSIZE, ell0)
elif args.attack == "anypixel":
    ell0 = int(ATK_PARAMS)
    infostring += "\nUsing general sparse rs attack with ell0 bound " + str(ell0)
    attackmodel = GeneralEll0BinaryColour(IMSIZE, ell0)
elif args.attack == "squiggle":
    split = ATK_PARAMS.split('-')
    boxsize = int(split[0])
    points = int(split[1])
    infostring += "\nUsing boxed squiggle with box size " + str(boxsize) + " and " + str(points) + " control points."
    attackmodel = BoxedSquiggle(IMSIZE, 0, points, boxsize)
elif args.attack == "maskedbezier":
    split = ATK_PARAMS.split('-')
    ell0 = int(split[0])
    width = int(split[1])
    count = int(split[2])
    infostring += "\nUsing masked bezier attack with " + str(count) + " lines, ell0 bound " + str(ell0) + " and width " + str(width)
    attackmodel = MaskedBezierAttack(IMSIZE, int(ell0/width), width, count, True)
elif args.attack == "cut-maskedbezier":
    split = ATK_PARAMS.split('-')
    ell0 = int(split[0])
    width = int(split[1])
    count = int(split[2])
    infostring += "\nUsing CUT masked bezier attack with " + str(count) + " lines, ell0 bound " + str(ell0) + " and width " + str(width)
    attackmodel = CutMaskedBezierAttack(IMSIZE, int(ell0/width), width, count, True)
elif args.attack == "maskedrs":
    ell0 = int(ATK_PARAMS)
    infostring += "\nUsing masked ell0 attack with ell0 " + str(ell0)
    attackmodel = MaskedEll0Attack(IMSIZE, ell0, True)
elif args.attack == "maskedpatch":
    split = ATK_PARAMS.split('-')
    boxsize = int(split[0])
    boxcount = int(split[1])
    infostring += "\nUsing masked patch attack with size: " + str(boxsize) + " and boxcount: " + str(boxcount)
    attackmodel = MaskedPatchAttack(IMSIZE, boxsize, boxcount, True)
elif args.attack == "patch":
    split = ATK_PARAMS.split('-')
    boxsize = int(split[0])
    boxcount = int(split[1])
    infostring += "\nUsing patch attack with size: " + str(boxsize) + " and boxcount: " + str(boxcount)
    attackmodel = PatchAttack(IMSIZE, boxsize, boxcount, True)
elif args.attack == "patchrs":
    if args.optimizer != "patchrs":
        raise ValueError("Invalid optimizer/attack combo")
    boxsize = int(ATK_PARAMS)
    infostring += "\nUsing patchrs with size: " + str(boxsize)
    attackmodel = NonParametricPatchAttack(IMSIZE, boxsize)
elif args.attack == "bezier_n":
    split = ATK_PARAMS.split('-')
    ell0 = int(split[0])
    width = int(split[1])
    count = int(split[2])
    infostring += "\nUsing torch bezier attack (inputting normalized images) with " + str(count) + " lines, ell0 bound " + str(ell0) + " and width " + str(width)
    attackmodel = TorchBezierAttack(IMSIZE, int(ell0/width), width, count)
elif args.attack == "frame":
    width = int(ATK_PARAMS)
    infostring += "\nUsing frame attack with thickness " + str(width)
    attackmodel = FrameAttack(IMSIZE, width, True)
else:
    print("Invalid attack model")
    exit(1)

# Optimizer to be used (in optimizer.py)
if args.optimizer == "rs":
    infostring += "\nUsing random search optimizer"
    # Gather schedule from optimizator params
    split = OPT_PARAMS.split('-')
    if split[0] == "0":
        schedule = [1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    else:
        schedule = []
        split = args.optparams.split('-')
        for i in range(10):
            schedule.append(float(split[i]))
    # schedule_iter = (np.array([0, 50, 200, 500, 1000, 2000, 4000, 6000, 8000]) * (MAX_ITER / 10000.0)).astype(int)
    schedule_iter = np.array([0, 50, 200, 500, 1000, 2000, 4000, 6000, 8000])
    schedule_factor = np.array(schedule)
    if ell0<=0:
        raise ValueError("Invalid ell0")
    optimizer = RandomSearch(attackmodel.searchspace, BATCH_SIZE, schedule_iter, schedule_factor, 1.0, ell0, attackmodel.pixel_feature_count())
elif args.optimizer == "ip":
    infostring += "\nUsing interpolation optimizer"
    # Gather schedule from optimizator params
    split = OPT_PARAMS.split('-')
    if split[0] == "0":
        schedule = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    else:
        schedule = []
        split = args.optparams.split('-')
        for i in range(10):
            schedule.append(float(split[i]))
    schedule_iter = (np.array([0, 50, 200, 500, 1000, 2000, 4000, 6000, 8000]) * (MAX_ITER / 10000.0)).astype(int)
    schedule_factor = np.array(schedule)
    optimizer = InterpolSearch(attackmodel.searchspace, BATCH_SIZE, schedule_iter, schedule_factor, 0.8)
elif args.optimizer == "de":
    split = OPT_PARAMS.split('-')
    popn = int(split[0])
    restarts = int(split[1])
    infostring += "\nUsing differential evolution optimizer with population " + str(popn) + " and restart every " + str(restarts) + " iterations."
    optimizer = DifferentialEvolution(attackmodel.searchspace, BATCH_SIZE, popn, restarts)
elif args.optimizer == "cma":
    infostring += "\nUsing CMA-ES optimizer"
    optimizer = CMAES(attackmodel.searchspace)
elif args.optimizer == "pso":
    infostring += "\nUsing particle swarm optimization (PSO) optimizer"
    optimizer = PSO(attackmodel.searchspace, BATCH_SIZE)
elif args.optimizer == "cut-pso":
    infostring += "\nUsing CUT particle swarm optimization (PSO) optimizer"
    optimizer = CutPSO(attackmodel.searchspace, BATCH_SIZE)
elif args.optimizer == "ngo":
    infostring += "\nUsing NGO optimizer"
    optimizer = NGO(attackmodel.searchspace, BATCH_SIZE, MAX_ITER)
elif args.optimizer == "cut-ngo":
    infostring += "\nUsing CutNGO optimizer"
    optimizer = CutNGO(attackmodel.searchspace, BATCH_SIZE, MAX_ITER)
elif args.optimizer == "patchrs":
    if args.attack != "patchrs":
        raise ValueError("Invalid optimizer/attack combo")
    infostring += "\nUsing patchrs optimizer"
    optimizer = NonParametricPatchSearch(attackmodel.searchspace, int(ATK_PARAMS), IMSIZE, BATCH_SIZE,
                                         [0, 10, 50, 200, 500, 1000, 2000, 12000], [12, 8, 6, 4, 3, 2, 1, 1], MAX_ITER)
else:
    print("Invalid optimizer")
    exit(1)

# assert TEST_SAMPLES > int(args.samples)

# Adversary is standard and should not be changed
adversary = Adversary(target, optimizer, loss, attackmodel)
optimizer.setadversary(adversary)

# ----------------------------------------------------------------------------------------------------------------------

# Load the sample image to perform the attack
# image_path = os.path.join(basepath, "../images/1.jpeg")
# classname_path = os.path.join(basepath, "../imagenet_classes.txt")

if DATASET == "imagenet":
    dataset_path = "/mnt/oberyn/lorisg96/imagenet/ILSVRC/Data/CLS-LOC/val"
    labels_file_path = "/mnt/oberyn/lorisg96/imagenet/ILSVRC/Annotations/valnew.txt"
    # classnames = open(classname_path).readlines()
elif DATASET == "gtsrb":
    dataset_path = os.path.join(basepath, "../datasets/gtsrb/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images")
    labels_file_path = os.path.join(basepath, "../datasets/gtsrb/test_labels.txt")
elif DATASET == "tsrd":
    dataset_path = os.path.join(basepath, "../datasets/tsrd/images")
    labels_file_path = os.path.join(basepath, "../datasets/tsrd/labels.txt")
elif DATASET == "masked_tsrd":
    dataset_path = os.path.join(basepath, "../datasets/tsrd_mask/test")
    mask_path = os.path.join(basepath, "../datasets/tsrd_mask/testmask")
    labels_file_path = os.path.join(basepath, "../datasets/tsrd_mask/labels.txt")
else:
    raise ValueError("Invalid dataset name")

output_path = os.path.join(basepath, args.output)  # "../experiments/experiment01.csv")

outputfile = open(output_path, 'w', newline='')
outputcsv = csv.writer(outputfile, delimiter=',')

images = sorted([f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))])
imagecount = len(images)

# Tests model to evaluate initial performance
labels_file = open(labels_file_path, 'r')
labels = labels_file.readlines()

test_set = np.zeros([SAMPLES, IMSIZE, IMSIZE, 3]).astype(np.uint8)
preds = np.zeros([SAMPLES, CLASS_COUNT])
for i in range(SAMPLES):
    # Load RGB image
    im = Image.open(os.path.join(dataset_path, images[i])).convert('RGB')
    im = utils.torch_center_crop_resize(im, IMSIZE)
    raw = np.array(im).astype(np.uint8)
    test_set[i, :, :, :] = raw

if USE_TORCH_IMAGES:
    test_set = torch.from_numpy((test_set.transpose([0, 3, 1, 2]) / 255.0).astype(np.float32)).cuda()
    for i in range(SAMPLES):
        preds[i, :] = target.predict(test_set[i:i+1, :, :, :]).cpu().detach().numpy()
else:
    for i in range(SAMPLES):
        preds[i, :] = target.predict(test_set[i:i+1, :, :, :])

correct = 0
for count in range(SAMPLES):
    x = np.argmax(preds[count, :])
    y = int(labels[count])

    if x == y:
        correct += 1


print("Clean performance: ", end='')
print(correct/SAMPLES)


exit()

outputcsv.writerow(['image name', 'target label', 'original_label', 'adversarial_label', 'queries', 'ell0', '0..n params'])

print("*"*30)
print("This run's parameters")
print("")
print(infostring)
print("*"*30)
count = 0

for count in range(0, SAMPLES, BATCH_SIZE):
    if count > SAMPLES > 0:
        break

    batch = np.zeros([BATCH_SIZE, IMSIZE, IMSIZE, IMCHANNELS]).astype(np.uint8)

    # Build image batch
    # Open and resize image (conversion to RGB is needed as some images are actually L images)
    for i in range(BATCH_SIZE):
        # Load RGB image
        image = images[count+i]
        im = Image.open(os.path.join(dataset_path, image)).convert('RGB')
        im = utils.torch_center_crop_resize(im, IMSIZE)
        batch[i, :, :, 0:3] = np.array(im).astype(np.uint8)

        # If its a masked dataset, also load the mask in the fourth channel
        if IMCHANNELS == 4:
            mask = Image.open(os.path.join(mask_path, image))
            mask = utils.center_crop_resize(mask, IMSIZE)
            raw = np.array(mask).astype(np.uint8)
            batch[i, :, :, 3] = raw

        # imarray = batch[i, :, :, 0:3]
        # imarray[raw==0]=0
        # Image.fromarray(imarray).show()
        # #exit()

    if USE_TORCH_IMAGES:
        data = torch.from_numpy((batch.transpose([0, 3, 1, 2]) / 255.0).astype(np.float32)).cuda()
    else:
        data = batch.copy()

    result, queries, params = adversary.performattack(data, maxqueries=MAX_ITER)

    originalresults = target.predict(batch[:, :, :, 0:3])

    for i in range(BATCH_SIZE):
        originalidx = int(np.argmax(originalresults[i, :]))
        if result[i]:
            adversarialimage = attackmodel(batch[i:i+1, :, :, :], params[i:i+1, :]).reshape([1, IMSIZE, IMSIZE, 3])
            adversarialidx = int(np.argmax(target.predict(adversarialimage)[0]))
            ell0diff = utils.pixel_ell0_norm(adversarialimage, batch[i, :, :, 0:3])
            # -------------------------------------------
            if DEBUG:
                print("Target ", end='')
                print(labels[count+i].strip("\n"))
                print("Original ", end='')
                print(originalidx)
                print("Adversarial ", end='')
                print(adversarialidx)
                print("Iterations ", end='')
                print(queries[i])
                print("L0 ", end='')
                print(ell0diff)
                Image.fromarray(adversarialimage).show()
                input("wait ")
            # -------------------------------------------
        else:
            adversarialidx = -1
            ell0diff = -1
        # originallabel = classnames[originalidx]
        # adversariallabel = classnames[adversarialidx]

        outputcsv.writerow(
            [images[count+i], labels[count+i].strip("\n"), originalidx, adversarialidx, queries[i], ell0diff] + list(params[i, :]))
        outputfile.flush()

        print("\r Image " + str(count+i) + " of " + str(SAMPLES if SAMPLES > 0 else imagecount), end='')
