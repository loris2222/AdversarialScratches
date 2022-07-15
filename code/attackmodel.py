from abc import ABC, abstractmethod
import numpy as np
import utils
import math
from PIL import ImageDraw, Image
import torch
import matplotlib.pyplot as plt

class AttackModel(ABC):
    def __init__(self, imsize):
        self.searchspace = None
        self.imsize = float(imsize)
        pass

    @abstractmethod
    # Must return the grouping size of feature for every pixel, for example if a pixel is identified by x,y,r,g,b it
    # returns 5. If they are not grouped it must raise ValueError
    def pixel_feature_count(self) -> int:
        pass

    @abstractmethod
    # Given a numpy image returns the modified image given the parameters of the attack
    def __call__(self, sample, params):
        pass

    def get_valid_params(self, param_batch: list, sample_batch: list):
        return param_batch


class BezierAttack(AttackModel):
    def __init__(self, imsize=224, ell0=0, width=0, linecount=1, bynarize=True, degree=2):
        super().__init__(imsize)
        self.ell0 = ell0
        self.width = width
        self.linecount = linecount
        self.degree = degree
        if not bynarize:
            self.searchspace = self.linecount * ((self.degree + 1) * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0)] +
                                                 [(0.0, 255.0), (0.0, 255.0), (0.0, 255.0)])
        else:
            self.searchspace = self.linecount * ((self.degree + 1) * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0)] +
                                                 [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
        self.bynarize = 1 if bynarize else 0

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros(sample_batch.shape).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, :]
            params = params_batch[i, :]

            result = np.array(sample)
            offset = 0  # When drawing multiple lines, they are grouped by offset groups
            for line in range(self.linecount):
                xyz = [(params[offset + inner_offset], params[offset + inner_offset + 1]) for inner_offset in range(0, 2 * (self.degree + 1), 2)]
                color = (params[offset + 2 * (self.degree + 1) + 0] * (1 + 254 * self.bynarize),
                         params[offset + 2 * (self.degree + 1) + 1] * (1 + 254 * self.bynarize),
                         params[offset + 2 * (self.degree + 1) + 2] * (1 + 254 * self.bynarize))
                result = utils.superimpose_bezier_ell0(result, xyz, color, self.ell0, self.width)
                offset += 3 + 2 * (self.degree + 1)
            outputs[i, :, :, :] = result
            # plt.imshow(outputs[i])
            # plt.savefig("testattack")
            # exit()

        return outputs


class SplineAttack(AttackModel):
    def __init__(self, imsize=224, degree=2, ell0=0, width=0, linecount=1, bynarize=True):
        super().__init__(imsize)
        self.ell0 = ell0
        self.width = width
        self.linecount = linecount
        self.degree = degree
        if not bynarize:
            self.searchspace = self.linecount * ((self.degree + 1) * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0)] +
                                                 [(0.0, 255.0), (0.0, 255.0), (0.0, 255.0)])
        else:
            self.searchspace = self.linecount * ((self.degree + 1) * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0)] +
                                                 [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
        self.bynarize = 1 if bynarize else 0

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros(sample_batch.shape).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, :]
            params = params_batch[i, :]

            result = np.array(sample)
            offset = 0  # When drawing multiple lines, they are grouped by offset groups
            for line in range(self.linecount):
                xyz = [(params[offset + inner_offset], params[offset + inner_offset + 1]) for inner_offset in range(0, 2 * (self.degree + 1), 2)]
                color = (params[offset + 2 * (self.degree + 1) + 0] * (1 + 254 * self.bynarize),
                         params[offset + 2 * (self.degree + 1) + 1] * (1 + 254 * self.bynarize),
                         params[offset + 2 * (self.degree + 1) + 2] * (1 + 254 * self.bynarize))
                result = utils.superimpose_autodegree_spline_ell0(result, xyz, color, self.ell0, self.width)
                offset += 3 + 2 * (self.degree + 1)
            outputs[i, :, :, :] = result
            # plt.imshow(outputs[i])
            # plt.savefig("testattack")
            # exit()

        return outputs


# All Bézier in this attack will have the same color
class OneColorBezierAttack(AttackModel):
    def __init__(self, imsize=224, ell0=0, width=0, linecount=1, bynarize=True):
        super().__init__(imsize)
        self.ell0 = ell0
        self.width = width
        self.linecount = linecount
        if not bynarize:
            self.searchspace = self.linecount * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0)] \
                                                 + [(0.0, 255.0), (0.0, 255.0), (0.0, 255.0)]
        else:
            self.searchspace = self.linecount * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0)] \
                                                 + [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        self.bynarize = 1 if bynarize else 0

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros(sample_batch.shape).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, :]
            params = params_batch[i, :]

            result = np.array(sample)
            offset = 0  # When drawing multiple lines, they are grouped by offset groups
            for line in range(self.linecount):
                xyz = [(params[offset + 0], params[offset + 1]), (params[offset + 2], params[offset + 3]),
                       (params[offset + 4], params[offset + 5])]
                color = (params[self.linecount * 6 + 0] * (1 + 254 * self.bynarize),
                         params[self.linecount * 6 + 1] * (1 + 254 * self.bynarize),
                         params[self.linecount * 6 + 2] * (1 + 254 * self.bynarize))
                result = utils.superimpose_bezier_ell0(result, xyz, color, self.ell0, self.width)
                offset += 6
            outputs[i, :, :, :] = result
            # Image.fromarray(outputs[i]).show()
            # exit()

        return outputs


# Bézier color is taken with color picker from the image
class ImageColorBezierAttack(AttackModel):
    def __init__(self, imsize=224, ell0=0, width=0, linecount=1):
        super().__init__(imsize)
        self.ell0 = ell0
        self.width = width
        self.linecount = linecount
        self.searchspace = self.linecount * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                             (0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                             (0.0, self.imsize-1.0), (0.0, self.imsize-1.0)]
        self.bynarize = 0

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros(sample_batch.shape).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, :]
            params = params_batch[i, :]

            result = np.array(sample)
            offset = 0  # When drawing multiple lines, they are grouped by offset groups
            for line in range(self.linecount):
                xyz = [(params[offset + 0], params[offset + 1]), (params[offset + 2], params[offset + 3]),
                       (params[offset + 4], params[offset + 5])]
                color_xy = (params[offset + 6], params[offset + 7])
                color = sample[int(np.clip(color_xy[0], 0, self.imsize-1)), int(np.clip(color_xy[1], 0, self.imsize-1)), :]
                result = utils.superimpose_bezier_ell0(result, xyz, tuple(color), self.ell0, self.width)
                offset += 8
            outputs[i, :, :, :] = result
            # Image.fromarray(outputs[6]).show()
            # exit()

        return outputs


# Bezier attack with grayscale colors only
class GrayscaleBezierAttack(AttackModel):
    def __init__(self, imsize=224, ell0=0, width=0, linecount=1, bynarize=False):
        super().__init__(imsize)
        self.ell0 = ell0
        self.width = width
        self.linecount = linecount
        if not bynarize:
            self.searchspace = self.linecount * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, 255.0)]
        else:
            self.searchspace = self.linecount * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, 1.0)]
        self.bynarize = 1 if bynarize else 0

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros(sample_batch.shape).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, :]
            params = params_batch[i, :]

            result = np.array(sample)
            offset = 0  # When drawing multiple lines, they are grouped by offset groups
            for line in range(self.linecount):
                xyz = [(params[offset + 0], params[offset + 1]), (params[offset + 2], params[offset + 3]),
                       (params[offset + 4], params[offset + 5])]
                color = (params[offset + 6] * (1 + 254 * self.bynarize),
                         params[offset + 6] * (1 + 254 * self.bynarize),
                         params[offset + 6] * (1 + 254 * self.bynarize))
                result = utils.superimpose_bezier_ell0(result, xyz, color, self.ell0, self.width)
                offset += 7
            outputs[i, :, :, :] = result
            # Image.fromarray(outputs[i]).show()
            # exit()

        return outputs


class MulticolorBezierAttack(AttackModel):
    def __init__(self, imsize=224, ell0=0, width=0):
        super().__init__(imsize)
        self.ell0 = ell0
        self.width = width
        self.searchspace = [(0.0, self.imsize), (0.0, self.imsize),
                            (0.0, self.imsize), (0.0, self.imsize),
                            (0.0, self.imsize), (0.0, self.imsize)] \
                            + self.ell0 * [(0.0, 255.0), (0.0, 255.0), (0.0, 255.0)]

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample, params):
        raise ValueError  # Batching is not implemented
        params = np.array(params).astype(int)
        offset = 0
        xyz = [(params[offset + 0], params[offset + 1]), (params[offset + 2], params[offset + 3]),
               (params[offset + 4], params[offset + 5])]
        result = utils.superimpose_multicolor_bezier(sample, xyz, params[6:], self.ell0, self.width)
        return result


class BinaryColorBezierAttack(AttackModel):
    def __init__(self, imsize=224, ell0=0, width=0):
        super().__init__(imsize)
        self.ell0 = ell0
        self.width = width
        self.searchspace = [(0.0, self.imsize), (0.0, self.imsize),
                            (0.0, self.imsize), (0.0, self.imsize),
                            (0.0, self.imsize), (0.0, self.imsize)] \
                           + self.ell0 * [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample, params):
        raise ValueError  # Batching is not implemented
        params = np.around(np.array(params)).astype(int)
        offset = 0
        xyz = [(params[offset + 0], params[offset + 1]), (params[offset + 2], params[offset + 3]),
               (params[offset + 4], params[offset + 5])]
        result = utils.superimpose_multicolor_bezier(sample, xyz, params[6:] * 255, self.ell0, self.width)
        return result


class ConstrainedLengthStraightLine(AttackModel):
    def __init__(self, imsize, maxlength: float):
        super().__init__(imsize)
        self.maxlength = maxlength
        # center_x, center_y, length, theta, RGB
        self.searchspace = [(math.ceil(maxlength/2.0), self.imsize-math.ceil(maxlength/2.0)),
                            (math.ceil(maxlength/2.0), self.imsize-math.ceil(maxlength/2.0)),
                            (0.0, 360.0), (0.0, 255.0), (0.0, 255.0), (0.0, 255.0)]

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample, params):
        raise ValueError  # Batching is not implemented
        params = np.array(params).astype(int)
        xystart = (params[0], params[1])
        xyend = (params[0] + params[2]*math.cos(math.radians(params[3])), params[1] + params[2]*math.sin(math.radians(params[3])))
        xys = [xystart, xyend]
        color = (params[4], params[5], params[6])
        return utils.superimpose_line(sample, xys, color)


class FixedLengthStraightLine(AttackModel):
    def __init__(self, imsize, length: float):
        super().__init__(imsize)
        self.length = length
        # center_x, center_y, theta, RGB
        self.searchspace = [(math.ceil(length/2.0), self.imsize-math.ceil(length/2.0)),
                            (math.ceil(length/2.0), self.imsize-math.ceil(length/2.0)),
                            (0.0, 360.0), (0.0, 255.0), (0.0, 255.0), (0.0, 255.0)]

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample, params):
        raise ValueError  # Batching is not implemented
        params = np.array(params).astype(int)
        xystart = (params[0], params[1])
        xyend = (params[0] + self.length*math.cos(math.radians(params[2])), params[1] + self.length*math.sin(math.radians(params[2])))
        xys = [xystart, xyend]
        color = (params[3], params[4], params[5])
        return utils.superimpose_line(sample, xys, color)


class BoxedSquiggle(AttackModel):
    def __init__(self, imsize=224, ell0=0, points=10, boxsize=20):
        super().__init__(imsize)
        self.ell0 = ell0
        self.points = points
        self.boxsize = float(int(boxsize))
        # Space is composed of [box_x box_y squiggle_r squiggle_g squiggle_b point1_x point1_y ... pointn_x pointn_y]
        self.searchspace = [(0.0, self.imsize-self.boxsize), (0.0, self.imsize-self.boxsize)] + \
                           [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)] + \
                           self.points * [(0.0, self.boxsize), (0.0, self.boxsize)]

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros(sample_batch.shape).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, :]
            params = params_batch[i, :]

            boxxy = (params[0], params[1])
            color = (params[2]*255, params[3]*255, params[4]*255)

            im = Image.fromarray(sample)
            draw = ImageDraw.Draw(im)

            # compute two-tuple points for lines
            lines = [(params[j + 0] + boxxy[0], params[j + 1] + boxxy[1]) for j in range(5, 5+2*self.points, 2)]

            # restrict obtained line to ell0?
            if self.ell0 > 0:
                raise ValueError  # Ell 0 bound not implemented yet
                # TODO scan each line, compute ell0, make the sum and stop points when bound reached

            draw.line(lines, fill=color)
            outputs[i, :, :, :] = np.array(im)

        return outputs


class GeneralEll0Attack(AttackModel):
    def __init__(self, imsize=224, ell0=0, bynarize=False):
        super().__init__(imsize)
        self.ell0 = ell0
        # center_x, center_y, RGB
        if not bynarize:
            self.searchspace = self.ell0 * [(0.0, self.imsize), (0.0, self.imsize),
                                            (0.0, 255.0), (0.0, 255.0), (0.0, 255.0)]
        else:
            self.searchspace = self.ell0 * [(0.0, self.imsize), (0.0, self.imsize),
                                            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        self.searchspacesize = len(self.searchspace)
        self.bynarize = 1 if bynarize else 0

    def pixel_feature_count(self) -> int:
        return 5

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros(sample_batch.shape).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, :]
            params = params_batch[i, :]

            im = Image.fromarray(sample)
            draw = ImageDraw.Draw(im)

            points = [(params[j+0], params[j+1], params[j+2], params[j+3], params[j+4]) for j in range(0, self.searchspacesize, 5)]
            for point in points:
                draw.point([(point[0], point[1])],
                           # Get the original value if we are searching colors in [0,255], else multiply the color by
                           # 255 if we are searching in [0,1] so as to obtain only 0 or 255
                           fill=(point[2]*(1+254*self.bynarize),
                                 point[3]*(1+254*self.bynarize),
                                 point[4]*(1+254*self.bynarize)))

            outputs[i, :, :, :] = np.array(im)

        return outputs


class GeneralEll0BinaryColour(AttackModel):
    def __init__(self, imsize=224, ell0=0):
        super().__init__(imsize)
        self.ell0 = ell0
        # center_x, center_y, RGB
        self.searchspace = self.ell0 * [(0.0, self.imsize), (0.0, self.imsize), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        self.searchspacesize = len(self.searchspace)

    def pixel_feature_count(self) -> int:
        return 5

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros([sample_batch.shape[0], sample_batch.shape[1], sample_batch.shape[2], 3]).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, 0:3]
            params = params_batch[i, :]

            im = Image.fromarray(sample)
            draw = ImageDraw.Draw(im)

            points = [(params[j+0], params[j+1], params[j+2]*255, params[j+3]*255, params[j+4]*255) for j in range(0, self.searchspacesize, 5)]
            for point in points:
                draw.point([(point[0], point[1])], fill=(point[2], point[3], point[4]))

            outputs[i, :, :, :] = np.array(im)

        return outputs


class MaskedBezierAttack(AttackModel):
    def __init__(self, imsize=224, ell0=0, width=0, linecount=1, bynarize=False):
        super().__init__(imsize)
        self.ell0 = ell0
        self.width = width
        self.linecount = linecount
        if not bynarize:
            self.searchspace = self.linecount * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, 255.0), (0.0, 255.0), (0.0, 255.0)]
        else:
            self.searchspace = self.linecount * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        self.bynarize = 1 if bynarize else 0

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros([sample_batch.shape[0], sample_batch.shape[1], sample_batch.shape[2], 3]).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            #sample = sample_batch[i, :, :, :]
            sample = sample_batch[i, :, :, 0:3]
            mask = sample_batch[i, :, :, 3]
            params = params_batch[i, :]

            result = np.array(sample)
            offset = 0  # When drawing multiple lines, they are grouped by offset groups
            for line in range(self.linecount):
                xyz = [(params[offset + 0], params[offset + 1]), (params[offset + 2], params[offset + 3]),
                       (params[offset + 4], params[offset + 5])]
                color = (params[offset + 6] * (1 + 254 * self.bynarize),
                         params[offset + 7] * (1 + 254 * self.bynarize),
                         params[offset + 8] * (1 + 254 * self.bynarize))
                # result = utils.superimpose_masked_bezier_ell0(result, xyz, color, self.ell0, self.width)
                result = utils.superimpose_bezier_ell0(result, xyz, color, self.ell0, self.width)
                result[mask == 0] = sample[mask == 0]
                offset += 9
            outputs[i, :, :, :] = result[:, :, 0:3]
            # plt.imshow(outputs[i])
            # plt.savefig("testattack")
            # exit()

        return outputs


class CutMaskedBezierAttack(AttackModel):
    def __init__(self, imsize=224, ell0=0, width=0, linecount=1, bynarize=False):
        super().__init__(imsize)
        self.ell0 = ell0
        self.width = width
        self.linecount = linecount
        if not bynarize:
            self.searchspace = self.linecount * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, 255.0), (0.0, 255.0), (0.0, 255.0)]
        else:
            self.searchspace = self.linecount * [(0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, self.imsize-1.0), (0.0, self.imsize-1.0), (0.0, self.imsize-1.0),
                                                 (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        self.bynarize = 1 if bynarize else 0

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros([sample_batch.shape[0], sample_batch.shape[1], sample_batch.shape[2], 3]).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, 0:3]
            mask = sample_batch[i, :, :, 3]
            params = params_batch[i, :]
            #print(params)

            result = np.array(sample)
            offset = 0  # When drawing multiple lines, they are grouped by offset groups
            for line in range(self.linecount):
                xyz = [(params[offset + 0], params[offset + 1]), (params[offset + 2], params[offset + 3]),
                       (params[offset + 4], params[offset + 5])]
                color = (params[offset + 6] * (1 + 254 * self.bynarize),
                         params[offset + 7] * (1 + 254 * self.bynarize),
                         params[offset + 8] * (1 + 254 * self.bynarize))
                # result = utils.superimpose_bezier_ell0(result, xyz, color, self.ell0, self.width)
                result = utils.superimpose_bezier_ell0(result, xyz, color, 0, self.width)
                # result[mask == 0] = sample[mask == 0]
                offset += 9
            outputs[i, :, :, :] = result[:, :, 0:3]
        # testim = outputs[0, :, :, :]
        # testim[sample_batch[0, :, :, 3]==0]=0
        # im = Image.fromarray(testim)
        # im.show()

        return outputs

    # returns a list long len(sample_batch) where each row is a list of parameters
    def get_valid_params(self, params_batch_in, sample_batch: np.ndarray):
        # self.__call__(sample_batch, params_batch)
        batchsize = sample_batch.shape[0]
        params_batch_in = np.array(params_batch_in)
        params_batch = np.around(params_batch_in).astype(int)
        out_params = np.zeros_like(params_batch).astype(np.double)
        for i in range(batchsize):
            sample = sample_batch[i, :, :, 0:3]
            mask = sample_batch[i, :, :, 3]
            # sample[mask==0]=0
            # Image.fromarray(sample).show()
            params = params_batch[i, :]
            sample_out_params = list()

            offset = 0  # When drawing multiple lines, they are grouped by offset groups
            for line in range(self.linecount):
                xys = [(np.clip(params[offset + 0], 0, 223), np.clip(params[offset + 1], 0, 223)), (np.clip(params[offset + 2], 0, 223), np.clip(params[offset + 3], 0, 223)),
                       (np.clip(params[offset + 4], 0, 223), np.clip(params[offset + 5], 0, 223))]
                drawn = 0
                imsize = sample_batch.shape[1]
                ts = [t / float(imsize) for t in range(imsize + 1)]
                bezier = utils.make_bezier(xys)
                points = bezier(ts)
                start_draw = 0
                t0 = 0
                t2 = 0
                # for j in range(len(points) - 1):
                #     sample[math.floor(points[j][0]), math.floor(points[j][1])]=0
                # Image.fromarray(sample).show()
                for j in range(len(points) - 1):
                    diffx = abs(math.floor(points[j + 1][0]) - math.floor(points[j][0]))
                    diffy = abs(math.floor(points[j + 1][1]) - math.floor(points[j][1]))
                    todraw = max(diffx, diffy)
                    if start_draw == 0 and mask[math.floor(points[j][1])][math.floor(points[j][0])] == 1:
                        start_draw = 1
                        t0 = j / (len(points) - 1)
                    if start_draw == 1:
                        if mask[math.floor(points[j][1])][math.floor(points[j][0])] == 0 or drawn >= self.ell0 or j == len(points) - 2:
                            t2 = j / (len(points) - 1)
                            new_params = self.split_bezier(xys, t0, t2)
                            new_params.extend([params_batch_in[i, offset + 6], params_batch_in[i, offset + 7], params_batch_in[i, offset + 8]])
                            sample_out_params.extend(new_params)
                            break
                        else:
                            drawn += todraw
                    if j == len(points) - 2 and start_draw == 0:
                        new_params = self.split_bezier(xys, t0, t2)
                        new_params.extend([params_batch_in[i, offset + 6], params_batch_in[i, offset + 7], params_batch_in[i, offset + 8]])
                        sample_out_params.extend(new_params)
                        break

                offset += 9
            out_params[i, :] = np.array(sample_out_params)
        return out_params

    def bezier(self, p0, p1, p2, t):
        x = (1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * p1[1] + t ** 2 * p2[1]
        return x, y

    def split_bezier(self, xys, t0, t2):
        P0 = np.array([xys[0][0], xys[0][1]])
        P1 = np.array([xys[1][0], xys[1][1]])
        P2 = np.array([xys[2][0], xys[2][1]])
        PA = np.array(self.bezier(P0, P1, P2, t0))
        P1_P = (1 - t0) * P1 + t0 * P2
        PB = np.array(self.bezier(P0, P1, P2, t2))
        P2_P = (1 - t2) * PA + t2 * P1_P
        return [float(PA[0]), float(PA[1]), float(P2_P[0]), float(P2_P[1]), float(PB[0]), float(PB[1])]


class MaskedEll0Attack(AttackModel):
    def __init__(self, imsize=224, ell0=0, bynarize=False):
        super().__init__(imsize)
        self.ell0 = ell0
        # center_x, center_y, RGB
        if not bynarize:
            self.searchspace = self.ell0 * [(0.0, self.imsize), (0.0, self.imsize),
                                            (0.0, 255.0), (0.0, 255.0), (0.0, 255.0)]
        else:
            self.searchspace = self.ell0 * [(0.0, self.imsize), (0.0, self.imsize),
                                            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        self.searchspacesize = len(self.searchspace)
        self.bynarize = 1 if bynarize else 0

    def pixel_feature_count(self) -> int:
        return 5

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros([sample_batch.shape[0], sample_batch.shape[1], sample_batch.shape[2], 3]).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            image = sample_batch[i, :, :, 0:3]
            mask = sample_batch[i, :, :, 3]
            params = params_batch[i, :]

            im = Image.fromarray(image)
            draw = ImageDraw.Draw(im)

            points = [(params[j+0], params[j+1], params[j+2], params[j+3], params[j+4]) for j in range(0, self.searchspacesize, 5)]
            for point in points:
                if mask[point[0], point[1]] > 0:
                    draw.point([(point[0], point[1])],
                               # Get the original value if we are searching colors in [0,255], else multiply the color
                               # by 255 if we are searching in [0,1] so as to obtain only 0 or 255
                               fill=(point[2]*(1+254*self.bynarize),
                                     point[3]*(1+254*self.bynarize),
                                     point[4]*(1+254*self.bynarize)))

            outputs[i, :, :, :] = np.array(im)

        return outputs


class PatchAttack(AttackModel):
    def __init__(self, imsize=224, boxsize=20, boxcount=15, bynarize=False):
        super().__init__(imsize)
        self.boxsize = boxsize
        self.boxcount = boxcount
        # center_x, center_y, RGB
        if not bynarize:
            self.searchspace = [(0.0, self.imsize-self.boxsize-1.0)] + [(0.0, self.imsize-self.boxsize-1.0)] + \
                               [(0.0, 255.0), (0.0, 255.0), (0.0, 255.0)] + \
                               self.boxcount * [(0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, 255.0), (0.0, 255.0), (0.0, 255.0)]
        else:
            self.searchspace = [(0.0, self.imsize-self.boxsize-1.0)] + [(0.0, self.imsize-self.boxsize-1.0)] + \
                               [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)] + \
                               self.boxcount * [(0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        self.searchspacesize = len(self.searchspace)
        self.bynarize = 1 if bynarize else 0

    def pixel_feature_count(self) -> int:
        return 5

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros(sample_batch.shape).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, :]
            params = params_batch[i, :]

            im = Image.fromarray(sample)
            draw = ImageDraw.Draw(im)

            # Compute rectangles from parameters
            base = (params[0], params[1])
            rectangles = [(np.clip(params[j + 0], 0, self.boxsize - 1.0).astype(int),
                           np.clip(params[j + 1], 0, self.boxsize - 1.0).astype(int),
                           np.clip(params[j + 0] + params[j + 2], 0, self.boxsize - 1.0).astype(int),
                           np.clip(params[j + 1] + params[j + 3], 0, self.boxsize - 1.0).astype(int),
                           np.clip(params[j + 4], 0, self.boxsize - 1.0).astype(int),
                           np.clip(params[j + 5], 0, self.boxsize - 1.0).astype(int),
                           np.clip(params[j + 6], 0, self.boxsize - 1.0).astype(int))
                          for j in range(5, self.searchspacesize, 7)]

            # Draw background
            draw.rectangle(
                [(base[0], base[1]), (base[0] + self.boxsize - 1, base[1] + self.boxsize - 1)],
                fill=(params[2] * (1 + 254 * self.bynarize),
                      params[3] * (1 + 254 * self.bynarize),
                      params[4] * (1 + 254 * self.bynarize)))

            # Draw overlay
            for rectangle in rectangles:
                draw.rectangle([(base[0] + rectangle[0], base[1] + rectangle[1]),
                                (base[0] + rectangle[2], base[1] + rectangle[3])],
                               fill=(rectangle[4] * (1 + 254 * self.bynarize),
                                     rectangle[5] * (1 + 254 * self.bynarize),
                                     rectangle[6] * (1 + 254 * self.bynarize)))

            outputs[i, :, :, :] = np.array(im)

        return outputs


class MaskedPatchAttack(AttackModel):
    def __init__(self, imsize=224, boxsize=20, boxcount=15, bynarize=False):
        super().__init__(imsize)
        self.boxsize = boxsize
        self.boxcount = boxcount
        # center_x, center_y, RGB
        if not bynarize:
            self.searchspace = [(0.0, self.imsize-self.boxsize-1.0)] + [(0.0, self.imsize-self.boxsize-1.0)] + \
                               [(0.0, 255.0), (0.0, 255.0), (0.0, 255.0)] + \
                               self.boxcount * [(0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, 255.0), (0.0, 255.0), (0.0, 255.0)]
        else:
            self.searchspace = [(0.0, self.imsize-self.boxsize-1.0)] + [(0.0, self.imsize-self.boxsize-1.0)] + \
                               [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)] + \
                               self.boxcount * [(0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, self.boxsize-1.0),
                                                (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        self.searchspacesize = len(self.searchspace)
        self.bynarize = 1 if bynarize else 0

    def pixel_feature_count(self) -> int:
        return 5

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros([sample_batch.shape[0], sample_batch.shape[1], sample_batch.shape[2], 3]).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            image = sample_batch[i, :, :, 0:3]
            mask = sample_batch[i, :, :, 3]
            params = params_batch[i, :]

            im = Image.fromarray(image)
            draw = ImageDraw.Draw(im)

            # Compute rectangles from parameters
            base = (params[0], params[1])
            rectangles = [(np.clip(params[j+0], 0, self.boxsize-1.0).astype(int),
                           np.clip(params[j+1], 0, self.boxsize-1.0).astype(int),
                           np.clip(params[j+0]+params[j+2], 0, self.boxsize-1.0).astype(int),
                           np.clip(params[j+1]+params[j+3], 0, self.boxsize-1.0).astype(int),
                           np.clip(params[j+4], 0, self.boxsize-1.0).astype(int),
                           np.clip(params[j+5], 0, self.boxsize-1.0).astype(int),
                           np.clip(params[j+6], 0, self.boxsize-1.0).astype(int))
                          for j in range(5, self.searchspacesize, 7)]

            # Draw background
            draw.rectangle(
                [(base[0], base[1]), (base[0] + self.boxsize-1, base[1] + self.boxsize-1)],
                fill=(params[2] * (1 + 254 * self.bynarize),
                      params[3] * (1 + 254 * self.bynarize),
                      params[4] * (1 + 254 * self.bynarize)))

            # Draw overlay
            for rectangle in rectangles:
                draw.rectangle([(base[0]+rectangle[0], base[1]+rectangle[1]), (base[0]+rectangle[2], base[1]+rectangle[3])],
                               fill=(rectangle[4] * (1 + 254 * self.bynarize),
                                     rectangle[5] * (1 + 254 * self.bynarize),
                                     rectangle[6] * (1 + 254 * self.bynarize)))

            image[mask > 0] = np.array(im)[mask > 0]
            outputs[i, :, :, :] = image
            # plt.imshow(outputs[i])
            # plt.savefig("testattack")
            # exit()

        return outputs


class NonParametricPatchAttack(AttackModel):
    def __init__(self, imsize=224, boxsize=20):
        super().__init__(imsize)
        self.boxsize = boxsize
        # center_x, center_y, RGB
        self.searchspace = [(0.0, self.imsize - self.boxsize - 1.0)] + [(0.0, self.imsize - self.boxsize - 1.0)] + \
                           [(0.0, 1.0)] * 3 * (self.boxsize * self.boxsize)

        self.searchspacesize = len(self.searchspace)

    def pixel_feature_count(self) -> int:
        return 5

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros(sample_batch.shape).astype(np.uint8)
        params_batch = np.around(np.array(params_batch).astype(float)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, :]
            params = params_batch[i, :]

            im = Image.fromarray(sample)
            draw = ImageDraw.Draw(im)

            for k in range(0, self.boxsize*self.boxsize):
                draw.point((params[0] + k % self.boxsize, params[1] + math.floor(k/self.boxsize)),
                           fill=(params[3*k+2]*255, params[3*k+3]*255, params[3*k+4]*255))

            outputs[i, :, :, :] = np.array(im)

        return outputs


# Same as BezierAttack, with binary colours, and only for torch models, so does not pass through PIL images
# Torch images are Tensors in range [0, 1] and are [channels, x, y]
class TorchBezierAttack(AttackModel):
    def __init__(self, imsize=224, ell0=0, width=0, linecount=1):
        super().__init__(imsize)
        self.ell0 = ell0
        self.width = width
        self.linecount = linecount
        self.searchspace = self.linecount * [(0.0, self.imsize), (0.0, self.imsize), (0.0, self.imsize),
                                             (0.0, self.imsize), (0.0, self.imsize), (0.0, self.imsize),
                                             (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    def pixel_feature_count(self) -> int:
        raise ValueError

    def generate_bezier_mask(self, bezier_xys, ell0=0, width=0):
        mask = Image.fromarray(np.zeros([int(self.imsize), int(self.imsize), 3], dtype=np.uint8))
        draw = ImageDraw.Draw(mask)

        ts = [t / float(self.imsize) for t in range(int(self.imsize) + 1)]
        bezier = utils.make_bezier(bezier_xys)
        points = bezier(ts)

        if ell0 > 0:
            drawn = 0
            for i in range(len(points) - 1):
                diffx = abs(math.floor(points[i + 1][0]) - math.floor(points[i][0]))
                diffy = abs(math.floor(points[i + 1][1]) - math.floor(points[i][1]))
                todraw = max(diffx, diffy)
                drawn += todraw
                if drawn >= ell0:
                    break
                draw.line(points[i:i + 2], fill=(255, 255, 255), width=width)
        else:
            raise ValueError("Cannot generate bezier mask with unlimited ell0")

        # mask is a 3 channel version in format [x,y,c] of the mask. We need to binarize it and squish it
        maskimagearray = (np.array(mask)[:, :, 0]/255).astype(np.float)

        return maskimagearray

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = sample_batch.clone()
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = outputs[i, :, :, :]
            params = params_batch[i, :]

            offset = 0  # When drawing multiple lines, they are grouped by offset groups
            for line in range(self.linecount):
                xyz = [(params[offset + 0], params[offset + 1]), (params[offset + 2], params[offset + 3]),
                       (params[offset + 4], params[offset + 5])]
                color = torch.from_numpy(np.array([[params[offset + 6], params[offset + 7], params[offset + 8]]]).astype(np.float32).transpose()).cuda()
                mask = self.generate_bezier_mask(xyz, self.ell0, self.width)
                sample[:, mask > 0] = color
                offset += 9
            outputs[i, :, :, :] = sample

        return outputs


# Adversarial frame of width pixels
class FrameAttack(AttackModel):
    def __init__(self, imsize: int = 224, width: int = 1, bynarize=True):
        super().__init__(imsize)
        self.width = width
        self.imsize = imsize
        if not bynarize:
            self.searchspace = [(0.0, 255.0) for _ in range(3 * 2 * width * imsize)] \
                               + [(0.0, 255.0) for _ in range(3 * (imsize - 2 * width) * width)]
        else:
            self.searchspace = [(0.0, 1.0) for _ in range(3 * 2 * width * imsize)] \
                               + [(0.0, 1.0) for _ in range(3 * (imsize - 2 * width) * width * 2)]
        self.bynarize = 1 if bynarize else 0
        self.searchspacesize = len(self.searchspace)

    def pixel_feature_count(self) -> int:
        raise ValueError

    def __call__(self, sample_batch, params_batch):
        batchsize = sample_batch.shape[0]
        outputs = np.zeros([sample_batch.shape[0], sample_batch.shape[1], sample_batch.shape[2], 3]).astype(np.uint8)
        params_batch = np.around(np.array(params_batch)).astype(int)

        for i in range(batchsize):
            sample = sample_batch[i, :, :, 0:3]
            params = params_batch[i, :]

            result = np.array(sample)
            idx = 0
            # Draw top lines
            for y in range(self.width):
                for x in range(self.imsize):
                    result[x, y, :] = [params[idx + 0] * (1 + 254 * self.bynarize),
                                       params[idx + 1] * (1 + 254 * self.bynarize),
                                       params[idx + 2] * (1 + 254 * self.bynarize)]
                    idx += 3

            # Draw bottom lines
            for y in range(self.imsize - self.width, self.imsize):
                for x in range(self.imsize):
                    result[x, y, :] = [params[idx + 0] * (1 + 254 * self.bynarize),
                                       params[idx + 1] * (1 + 254 * self.bynarize),
                                       params[idx + 2] * (1 + 254 * self.bynarize)]
                    idx += 3

            # Draw left columns
            for y in range(self.width, self.imsize - self.width):
                for x in range(self.width):
                    result[x, y, :] = [params[idx + 0] * (1 + 254 * self.bynarize),
                                       params[idx + 1] * (1 + 254 * self.bynarize),
                                       params[idx + 2] * (1 + 254 * self.bynarize)]
                    idx += 3

            # Draw right columns
            for y in range(self.width, self.imsize - self.width):
                for x in range(self.imsize - self.width, self.imsize):
                    result[x, y, :] = [params[idx + 0] * (1 + 254 * self.bynarize),
                                       params[idx + 1] * (1 + 254 * self.bynarize),
                                       params[idx + 2] * (1 + 254 * self.bynarize)]
                    idx += 3
            outputs[i, :, :, :] = result
            Image.fromarray(result).show()
            exit()

        return outputs
