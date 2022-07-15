from PIL import Image, ImageDraw
import numpy as np
import os
import math
from operator import itemgetter
import torchvision.transforms as transforms
from scipy.interpolate import splev


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()


def pascal_row(n, memo={}):
    # This returns the nth row of Pascal's Triangle
    if n in memo:
        return memo[n]
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    memo[n] = result
    return result


def make_bezier(xys):
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n-1)

    def bezier(ts):
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier


# Superimposes a spline whose degree is L-2 where L is the number of control points
# Points in format [(x,y), (x,y)]
def superimpose_autodegree_spline_ell0(np_image, spline_xys, colour, ell0=0, width=0):
    spline_knots = np.array(spline_xys)
    knot_x = spline_knots[:, 0]
    knot_y = spline_knots[:, 1]

    knot_count = len(knot_x)
    spline_degree = knot_count - 1
    t = np.linspace(0, 1, knot_count-(spline_degree-1), endpoint=True)
    t = np.append([0]*spline_degree, t)
    t = np.append(t, [1]*spline_degree)
    tck = [t, [knot_x, knot_y], spline_degree]
    u3 = np.linspace(0, 1, ell0*2, endpoint=True)
    points = splev(u3, tck)
    points = [(points[0][i], points[1][i]) for i in range(len(points[0]))]

    im = Image.fromarray(np_image.astype(np.uint8))
    imsize = (np_image.shape[0], np_image.shape[1])
    draw = ImageDraw.Draw(im)

    if ell0 > 0:
        drawn = 0
        for i in range(len(points)-1):
            diffx = abs(math.floor(points[i+1][0])-math.floor(points[i][0]))
            diffy = abs(math.floor(points[i+1][1])-math.floor(points[i][1]))
            todraw = max(diffx, diffy)
            drawn += todraw
            if drawn >= ell0:
                break
            draw.line(points[i:i+2], fill=colour, width=width)
    else:
        draw.line(points, fill=colour, width=width)
    return np.array(im)


def superimpose_bezier(np_image, bezier_xys, colour):
    im = Image.fromarray(np_image.astype(np.uint8))
    imsize = (np_image.shape[0], np_image.shape[1])
    draw = ImageDraw.Draw(im)

    ts = [t / float(imsize[0]) for t in range(imsize[0]+1)]
    bezier = make_bezier(bezier_xys)
    points = bezier(ts)
    draw.line(points, fill=colour)
    return np.array(im)


def superimpose_bezier_ell0(np_image, bezier_xys, colour, ell0=0, width=0):
    im = Image.fromarray(np_image.astype(np.uint8))
    imsize = (np_image.shape[0], np_image.shape[1])
    draw = ImageDraw.Draw(im)

    ts = [t / float(imsize[0]) for t in range(imsize[0]+1)]
    bezier = make_bezier(bezier_xys)
    points = bezier(ts)

    if ell0 > 0:
        drawn = 0
        for i in range(len(points)-1):
            diffx = abs(math.floor(points[i+1][0])-math.floor(points[i][0]))
            diffy = abs(math.floor(points[i+1][1])-math.floor(points[i][1]))
            todraw = max(diffx, diffy)
            drawn += todraw
            if drawn >= ell0:
                break
            draw.line(points[i:i+2], fill=colour, width=width)
    else:
        draw.line(points, fill=colour, width=width)
    return np.array(im)


def superimpose_masked_bezier_ell0(np_image, bezier_xys, colour, ell0=0, width=0):
    im = Image.fromarray(np_image[:, :, 0:3].astype(np.uint8))
    mask = np_image[:, :, 3:4]
    imsize = (np_image.shape[0], np_image.shape[1])
    draw = ImageDraw.Draw(im)

    ts = [t / float(imsize[0]) for t in range(imsize[0]+1)]
    bezier = make_bezier(bezier_xys)
    points = bezier(ts)

    if ell0 > 0:
        drawn = 0
        for i in range(len(points)-1):
            p1 = (math.floor(points[i][0]), math.floor(points[i][1]))
            p2 = (math.floor(points[i+1][0]), math.floor(points[i+1][1]))

            # Compute pixel size of segment to draw
            diffx = abs(p2[0]-p1[0])
            diffy = abs(p2[1]-p1[1])
            todraw = max(diffx, diffy)

            # Only draw it if it is inside the mask, and exit when ell0 limit reached
            if mask[p1[0], p1[1]] > 0 and mask[p2[0], p2[1]] > 0:
                drawn += todraw
                if drawn >= ell0:
                    break
                draw.line([p1, p2], fill=colour, width=width)
    else:
        raise ValueError("Ell0 must be greater than 0")
    return np.concatenate((im, mask), axis=2)


def superimpose_multicolor_bezier(np_image, bezier_xys, colours, ell0=0, width=0):
    im = Image.fromarray(np_image.astype(np.uint8))
    imsize = (np_image.shape[0], np_image.shape[1])
    draw = ImageDraw.Draw(im)

    ts = [t / float(imsize[0]) for t in range(imsize[0] + 1)]
    bezier = make_bezier(bezier_xys)
    points = bezier(ts)
    if ell0 > 0:
        points = points[0:ell0]
    # for point in points:
    # draw.point(point, fill=colour)
    tuple_colours = [(colours[i+0], colours[i+1], colours[i+2]) for i in range(0, len(colours), 3)]
    for i in range(len(tuple_colours)-1):
        draw.line(points[i:i+2], fill=tuple_colours[i], width=width)
    return np.array(im)


def superimpose_line(np_image, line_xy, colour):
    im = Image.fromarray(np_image.astype(np.uint8))
    draw = ImageDraw.Draw(im)
    draw.line(line_xy, fill=colour)
    return np.array(im)


def pixel_ell0_norm(image1: np.ndarray, image2: np.ndarray):
    diff = np.abs(image1 - image2)
    diff = np.sum(diff, axis=2)
    diff = diff.flatten()
    return diff[diff > 0].shape[0]


def center_crop_resize(image: Image, targetsize: int):
    width, height = image.size  # Get dimensions
    size = min(width, height)

    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom)).resize((targetsize, targetsize))
    return image


# TODO fix usages of this when the model is not using torch but keras (but this returns PIL images!)
def torch_center_crop_resize(image: Image, targetsize: int):
    transform_center_crop = transforms.Compose(
        [transforms.Resize(targetsize), transforms.CenterCrop(targetsize), transforms.ToTensor()])
    transformed = transform_center_crop(image)
    imarray = transformed.numpy()
    image = Image.fromarray(((imarray*255.0).transpose([1, 2, 0]).astype(np.uint8)))
    return image


def slice_by_index(lst, indexes):
    if not lst or not indexes:
        return []
    slice_ = itemgetter(*indexes)(lst)
    if len(indexes) == 1:
        return [slice_]
    return list(slice_)


def test_simpledraw():
    imsize = (500, 500)

    im = Image.new('RGBA', imsize, (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    # Interpolation positions, 100 points equally spaced between 0 and 1
    ts = [t / 100.0 for t in range(101)]

    xys = [(0, 0), (0, 200), (240, 240)]
    bezier = make_bezier(xys)
    points = bezier(ts)

    # xys = [(100, 50), (100, 0), (50, 0), (50, 35)]
    # bezier = make_bezier(xys)
    # points.extend(bezier(ts))

    draw.line([(0, 0), (imsize[0] - 1, 0), (imsize[0] - 1, imsize[1] - 1), (0, imsize[1] - 1), (0, 0)])
    draw.line(points)
    # raw.polygon(points, fill = 'red')
    im.show()

def dataset_mu_sigma(path, imsize):
    import torch
    import torchvision.datasets as datasets
    import torch.utils.data as data

    imagenet = datasets.ImageFolder(path,
                                  transforms.Compose([
                                    transforms.Resize(imsize),
                                    transforms.CenterCrop(imsize),
                                    transforms.ToTensor()
                                    ]))

    test_loader = data.DataLoader(imagenet, batch_size=100, shuffle=False, num_workers=0)

    """Compute the mean and sd in an online fashion

            Var[x] = E[X^2] - E^2[X]
        """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in test_loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


# Computes the bounding box of the different pixels between the two images
def perturb_area(im1, im2):
    imsize = im1.shape[0]

    minx = imsize
    miny = imsize
    maxx = 0
    maxy = 0

    for i in range(imsize):
        for j in range(imsize):
            t = im1[i, j, :] == im2[i, j, :]
            if not t.all():
                minx = min(minx, i)
                miny = min(miny, j)
                maxx = max(minx, i)
                maxy = max(miny, j)

    return (maxx-minx)*(maxy-miny)