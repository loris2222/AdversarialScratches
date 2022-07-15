from abc import ABC, abstractmethod
import numpy as np


class Perturbation(ABC):

    def __init__(self, searchspace: list):
        # Search space is a list of 2-tuples that indicate min and max values for every dimension in the space
        self.searchspace = searchspace
        self.paramcount = len(self.searchspace)

    @abstractmethod
    # Re-initializes the optimizer's state for the following search.
    def __call__(self, point, batchsize):
        pass


# Computes independed gaussian perturbation on all parameters based on the sigma and mus defined at init
class GaussianPerturbation(Perturbation):

    def __init__(self, searchspace: list, sigma: np.ndarray, mu: np.ndarray = None):
        super().__init__(searchspace)
        self.sigma = sigma
        self.mu = mu
        self.pointshape = np.zeros([1, self.paramcount]).shape
        if self.mu is None:
            self.mu = np.zeros(self.paramcount)

        if self.sigma.shape != np.zeros(self.paramcount).shape:
            raise ValueError("Sigma must be accounted for all dimensions in the search space, received: "+str(self.sigma.shape)+" expected: "+str(np.zeros(self.paramcount).shape))
        if self.mu.shape != np.zeros(self.paramcount).shape:
            raise ValueError("Mu must be accounted for all dimensions in the search space, received: "+str(self.mu.shape)+" expected: "+str(np.zeros(self.paramcount).shape))

    def __call__(self, point: np.ndarray, batchsize=50):
        if point.shape != self.pointshape:
            print("Point shape is: "+str(point.shape))
            print("Expected: "+str(self.pointshape))
            raise ValueError("Point to be perturbed should not be flattened")

        ret = np.zeros([batchsize, self.paramcount])
        for sample in range(batchsize):
            count = 0
            for param in self.searchspace:
                # Perturb and clamp
                ret[sample, count] = point[0, count] + np.random.normal(self.mu[count], self.sigma[count])
                ret[sample, count] = np.clip(ret[sample, count], param[0], param[1])
                count += 1

        return ret


# Computes rigid motion on parameters. Requires specifying which parameters relate to a x coordinate (set 1 in list) and
# which to a y coordinate (set -1 in list), while ignoring all other parameters (set 0 in list). Because we still need
# to clamp values, rigid motion will be squashed against range bound if necessary.
class RigidMotion(Perturbation):

    def __init__(self, searchspace: list, enable: list, sigma: float):
        super().__init__(searchspace)
        self.sigma = sigma
        self.enable = enable
        self.pointshape = np.zeros([1, self.paramcount]).shape

    def __call__(self, point: np.ndarray, batchsize=50):
        if point.shape != self.pointshape:
            print("Point shape is: "+str(point.shape))
            print("Expected: "+str(self.pointshape))
            raise ValueError("Point to be perturbed should not be flattened")

        ret = np.zeros([batchsize, self.paramcount])
        for sample in range(batchsize):
            count = 0
            perturb_x = np.random.normal(0.0, self.sigma)
            perturb_y = np.random.normal(0.0, self.sigma)
            for param in self.searchspace:
                # Perturb and clamp
                if self.enable[count] == 1:
                    ret[sample, count] = point[0, count] + perturb_x
                    ret[sample, count] = np.clip(ret[sample, count], param[0], param[1])
                elif self.enable[count] == -1:
                    ret[sample, count] = point[0, count] + perturb_y
                    ret[sample, count] = np.clip(ret[sample, count], param[0], param[1])
                else:
                    ret[sample, count] = point[0, count]
                count += 1

        return ret


# Computes rigid motion on parameters. Requires specifying which parameters relate to a x coordinate (set 1 in list) and
# which to a y coordinate (set -1 in list), while ignoring all other parameters (set 0 in list). Because we still need
# to clamp values, rigid motion will be squashed against range bound if necessary.
class RigidAbsoluteGrid(Perturbation):

    def __init__(self, searchspace: list, enable: list, sigma: float):
        super().__init__(searchspace)
        self.sigma = sigma
        self.enable = enable
        self.pointshape = np.zeros([1, self.paramcount]).shape

    def __call__(self, point: np.ndarray, batchsize=64):
        if point.shape != self.pointshape:
            print("Point shape is: "+str(point.shape))
            print("Expected: "+str(self.pointshape))
            raise ValueError("Point to be perturbed should not be flattened")

        if batchsize != 64:
            # TODO use gmpy.is_square() to check if it's a perfect square
            raise ValueError("Different batch sizes not implemented")
        else:
            # TODO compute it as sqrt of batchsize
            gridcount = 8

        # Compute envelope of all features by scanning features marked as 1 and -1 and taking min/max
        # TODO can't compute envelope in this way, you must change how you do it (possibly comparing the perturbed
        #  image, but could also be by just selecting some point and offsetting. not recommended though.
        #  if it's limited to squiggles, however, that would be easier as the envelope is already defined in the space
        envelope = [[0.0, 0.0], [0.0, 0.0]]
        # envelope[coordinate][0=min, 1=max]
        envelope[0][0] = 224.0
        envelope[0][1] = 0.0
        envelope[1][0] = 224.0
        envelope[1][1] = 0.0
        count = 0
        for param in range(self.paramcount):
            if self.enable[count] == 1:
                envelope[0][0] = min(point[0, param], envelope[0][0])
                envelope[0][1] = max(point[0, param], envelope[0][1])
            elif self.enable[count] == -1:
                envelope[1][0] = min(point[0, param], envelope[1][0])
                envelope[1][1] = max(point[0, param], envelope[1][1])
            count += 1

        # Offset the point so that the top left of its envelope is placed at (0,0)
        count = 0
        zero_point = point.copy()
        for _ in range(self.paramcount):
            if self.enable[count] == 1:
                zero_point[0, count] = point[0, count] - envelope[0][0]
            elif self.enable[count] == -1:
                zero_point[0, count] = point[0, count] - envelope[1][0]
            else:
                zero_point[0, count] = point[0, count]
            count += 1

        # Create the perturb_x and perturb_y for all iterations so that they scan the whole grid
        slack = (224-(envelope[0][1]-envelope[0][0]), 224-(envelope[1][1]-envelope[1][0]))
        perturb_x = [float(i) * (slack[0]/float(gridcount)) for i in range(gridcount)]
        perturb_y = [float(i) * (slack[1]/float(gridcount)) for i in range(gridcount)]
        print(envelope)
        print(perturb_x)
        print(perturb_y)

        ret = np.zeros([batchsize, self.paramcount])
        for sample in range(batchsize):
            count = 0
            for param in self.searchspace:
                # Perturb and clamp
                if self.enable[count] == 1:
                    ret[sample, count] = zero_point[0, count] + perturb_x[sample % gridcount]
                elif self.enable[count] == -1:
                    ret[sample, count] = zero_point[0, count] + perturb_y[sample // gridcount]
                else:
                    ret[sample, count] = zero_point[0, count]
                count += 1

        return ret


# Creates offsets at equal distance around the centre for a squiggle
class SquiggleRelativeGrid(Perturbation):

    def __init__(self, searchspace: list, enable: list, stepsize: int, gridcount: int, patchsize: int):
        super().__init__(searchspace)
        self.stepsize = stepsize
        self.gridcount = gridcount
        self.enable = enable
        self.patchsize = patchsize
        self.pointshape = np.zeros([1, self.paramcount]).shape

    def __call__(self, point: np.ndarray, batchsize=64):
        if point.shape != self.pointshape:
            print("Point shape is: "+str(point.shape))
            print("Expected: "+str(self.pointshape))
            raise ValueError("Point to be perturbed should not be flattened")

        perturb = [np.longdouble(i) for i in range(int((-self.gridcount / 2) * self.stepsize), int((self.gridcount / 2) * self.stepsize), self.stepsize)]
        ret = np.zeros([batchsize, self.paramcount])
        for sample in range(batchsize):
            count = 0
            for param in self.searchspace:
                # Perturb and clamp
                if self.enable[count] == 1:
                    ret[sample, count] = point[0, count] + perturb[sample % self.gridcount]
                    ret[sample, count] = np.clip(ret[sample, count], 0, 224-self.patchsize)
                elif self.enable[count] == -1:
                    ret[sample, count] = point[0, count] + perturb[sample // self.gridcount]
                    ret[sample, count] = np.clip(ret[sample, count], 0, 224-self.patchsize)
                else:
                    ret[sample, count] = point[0, count]
                count += 1

        return ret
