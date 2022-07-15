from abc import ABC, abstractmethod
import adversary
import random
import math
import numpy as np
import cma
import nevergrad as ng
import copy
from PIL import Image

SOLUTION_CUBE_RADIUS = 3.0

np.seterr(all='warn')
generator = np.random.default_rng(seed=0)


class Optimizer(ABC):

    def __init__(self, searchspace: list, batchsize: int):
        # Search space is a list of 2-tuples that indicate min and max values for every dimension in the space
        self.searchspace = searchspace
        self.paramcount = len(self.searchspace)
        self.adversary = None
        self.batchsize = batchsize
        self.completedidx = [False for _ in range(batchsize)]
        self.completed = 0

    def setadversary(self, adversary):
        self.adversary = adversary

    @abstractmethod
    # Re-initializes the optimizer's state for the following search.
    def initstep(self):
        pass

    @abstractmethod
    # Search step asks the adversary to perform attacks with whatever parameters the optimizer deems suitable, then it
    # receives back the model's results and the optimizer can update its state.
    # Search step must check for exception from adversary.attacktarget() as this will be thrown when the attack was
    # successful, therefore the optimizer must stop execution.
    def searchstep(self):
        pass

    # Checks whether all the parameters in the solution are within the search space bounds.
    # This function is used for nevergrad optimizers which require a callable to check it.
    def isvalidsolution(self, inst: ng.p.Instrumentation) -> bool:
        point = inst[0][0]

        count = 0
        for elem in self.searchspace:
            if elem[0] > point[count] > elem[1]:
                continue
            return False
        return True

    # Checks whether all the parameters in the solution are within the search space bounds.
    # This function is used for nevergrad optimizers which require a callable to check it.
    def isincube(self, inst: ng.p.Instrumentation) -> bool:
        global SOLUTION_CUBE_RADIUS
        point = inst[0][0]

        count = 0
        for _ in range(self.paramcount):
            if -1.0*SOLUTION_CUBE_RADIUS < point[count] < 1.0*SOLUTION_CUBE_RADIUS:
                continue
            return False
        return True


# RandomSearch starts from a random point, then iteratively performs interpolation between the current point and another
# random point. If the loss improves with the new point, then the new point is taken as the first point and the process
# is repeated. If the loss does not improve, the first point is kept and a new try is issued. Parameter epsilon tells
# how far in the interpolation the new point will be taken: 0 means no change, 1 means the new point is taken.
class RandomSearch(Optimizer):
    def __init__(self, searchspace: list, batchsize: int, schedule_iter, schedule_factor, alpha_init=0.3, k=150, pixel_feature_count=5):
        super().__init__(searchspace, batchsize)
        self.alpha_init = alpha_init
        self.k = k
        self.currentpoints = None
        self.currentlosses = math.inf
        self.initstep()
        self.schedule_iter = schedule_iter
        self.schedule_length = len(self.schedule_iter)
        self.schedule_factor = schedule_factor
        self.pixel_feature_count = pixel_feature_count
        self.iter = 0

    def gen_random_point(self):
        res = []
        for elem in self.searchspace:
            rand = random.uniform(elem[0], elem[1])
            res.append(rand)
        return res

    def initstep(self):
        self.currentpoints = None
        self.currentlosses = np.ones([self.batchsize])*np.inf
        self.completedidx = [False for _ in range(self.batchsize)]
        self.completed = 0
        self.iter = 0

    def searchstep(self):
        self.iter += 1
        print("\rIteration "+str(self.iter)+" attacked "+str(self.completed)+" of "+str(self.batchsize)+" Avg loss: "+str(np.average(self.currentlosses)), end='')
        # The first step is different as it is just a generation
        if self.currentpoints is None:
            self.currentpoints = [self.gen_random_point() for _ in range(self.batchsize)]
            try:
                # Get initial losses from all points, also, if some samples are fooled at the first iteration, save
                # them in self.completedidx. Then initialize self.completed by scanning completedidx.
                self.currentlosses, self.completedidx = self.adversary.attacktarget(self.currentpoints, range(self.batchsize))
                for i in range(self.batchsize):
                    if self.completedidx[i]:
                        self.completed += 1
            except adversary.EndExecution:
                pass
            finally:
                return

        # From the second step onwards, we perform search and check whether we get a better result.
        # Find at what point we are in the schedule
        schedule_idx = self.schedule_length
        for i in range(self.schedule_length):
            if self.schedule_iter[i] > self.iter:
                schedule_idx = i-1
                break

        # Compute how many points we have to change in the solution
        alpha = self.alpha_init / self.schedule_factor[schedule_idx]
        pixeldiff = int(np.around(self.k*alpha))
        print(" Schedule status:"+str(schedule_idx)+" Epsit:"+str(pixeldiff)+" "+"#"*10, end='')

        # For samples that were not yet fooled, construct the new points
        newpoints = np.zeros([self.batchsize-self.completed, self.paramcount])
        idxs = []

        count = 0
        for idx in range(self.batchsize):
            if self.completedidx[idx]:
                continue

            # Generate random samples from which to compute deltas
            genpoint = self.gen_random_point()

            # Copy current point to be delta-ed
            newpoint = np.copy(self.currentpoints[idx])

            # Sample a number equal to pixeldiff of elements to change from the solution
            paramidx_tochange = np.random.choice(int(self.paramcount / self.pixel_feature_count), pixeldiff, replace=False)

            for param_idx in paramidx_tochange:
                newpoint[param_idx * self.pixel_feature_count:(param_idx + 1) * self.pixel_feature_count] = \
                    genpoint[param_idx * self.pixel_feature_count:(param_idx + 1) * self.pixel_feature_count]

            # Update lists
            newpoints[count, :] = newpoint
            idxs.append(idx)
            count += 1

        # Send the new points to the adversary (exception raised when maxquery reached)
        try:
            newlosses, newfooled = self.adversary.attacktarget(newpoints, idxs)
        except adversary.EndExecution:
            return

        count = 0
        for idx in idxs:
            if self.currentlosses[idx] > newlosses[count]:
                self.currentlosses[idx] = newlosses[count]
                self.currentpoints[idx] = newpoints[count]
            if newfooled[count]:
                self.completedidx[idx] = True
                self.completed += 1
            count += 1


# InterpolSearch is a version of random search where instead of changing a smaller number of parameters each time, we
# change all the parameters by a smaller amount. This is because our Bézier models are parametric.
class InterpolSearch(Optimizer):
    def __init__(self, searchspace: list, batchsize: int, schedule_iter, schedule_factor, epsilon_init=0.8):
        super().__init__(searchspace, batchsize)
        self.epsilon_init = epsilon_init
        self.currentpoints = None
        self.currentlosses = math.inf
        self.initstep()
        self.schedule_iter = schedule_iter
        self.schedule_length = len(self.schedule_iter)
        self.schedule_factor = schedule_factor
        self.iter = 0

    def gen_random_point(self):
        res = []
        for elem in self.searchspace:
            rand = random.uniform(elem[0], elem[1])
            res.append(rand)
        return res

    def initstep(self):
        self.currentpoints = None
        self.currentlosses = np.ones([self.batchsize])*np.inf
        self.completedidx = [False for _ in range(self.batchsize)]
        self.completed = 0
        self.iter = 0

    def searchstep(self):
        self.iter += 1
        print("\rIteration "+str(self.iter)+" attacked "+str(self.completed)+" of "+str(self.batchsize)+" Avg loss: "+str(np.average(self.currentlosses)), end='')
        # The first step is different as it does not require interpolation
        if self.currentpoints is None:
            self.currentpoints = [self.gen_random_point() for _ in range(self.batchsize)]
            try:
                # Get initial losses from all points, also, if some samples are fooled at the first iteration, save
                # them in self.completedidx. Then initialize self.completed by scanning completedidx.
                self.currentlosses, self.completedidx = self.adversary.attacktarget(self.currentpoints, range(self.batchsize))
                for i in range(self.batchsize):
                    if self.completedidx[i]:
                        self.completed += 1
            except adversary.EndExecution:
                pass
            finally:
                return

        # From the second step onwards, we perform search and check whether we get a better result.
        # Find at what point we are in the schedule
        schedule_idx = self.schedule_length
        for i in range(self.schedule_length):
            if self.schedule_iter[i] > self.iter:
                schedule_idx = i - 1
                break

        # Compute interpolation factor from schedule
        epsilon = self.epsilon_init / self.schedule_factor[schedule_idx]
        print(" Schedule status:" + str(schedule_idx) + " Epsilon:" + str(epsilon) + " " + "#" * 10, end='')

        # For samples that were not yet fooled, construct the new points
        newpoints = np.zeros([self.batchsize - self.completed, self.paramcount])
        idxs = []

        count = 0
        for idx in range(self.batchsize):
            if self.completedidx[idx]:
                continue

            # Generate random samples from which to compute deltas
            genpoint = self.gen_random_point()

            # Create interpolated new point
            newpoint = np.array((1 - epsilon) * np.array(self.currentpoints[idx]) + epsilon * np.array(genpoint))

            # Update lists
            newpoints[count, :] = newpoint
            idxs.append(idx)
            count += 1

        # Send the new points to the adversary (exception raised when maxquery reached)
        try:
            newlosses, newfooled = self.adversary.attacktarget(newpoints, idxs)
        except adversary.EndExecution:
            return

        count = 0
        for idx in idxs:
            if self.currentlosses[idx] > newlosses[count]:
                self.currentlosses[idx] = newlosses[count]
                self.currentpoints[idx] = newpoints[count]
            if newfooled[count]:
                self.completedidx[idx] = True
                self.completed += 1
            count += 1


# Differential evolution optimizer
class DifferentialEvolution(Optimizer):
    def __init__(self, searchspace: list, batchsize: int, n=10, cr=0.9, f=0.8, restart=-1):
        super().__init__(searchspace, batchsize)  # Batch size greater than 1 not yet implemented
        self.cr = cr    # Cross ratio
        self.f = f      # Differential weight
        self.n = n      # Population size
        self.restart = restart
        self.currentpoints = None
        self.currentlosses = [[math.inf for _ in range(0, self.n)] for _ in range(self.batchsize)]
        self.iter = 0

    def gen_random_point(self):
        res = []
        for elem in self.searchspace:
            rand = random.uniform(elem[0], elem[1])
            res.append(rand)
        return res

    def initstep(self):
        self.iter = 0
        self.currentpoints = [[self.gen_random_point() for _ in range(0, self.n)] for _ in range(self.batchsize)]
        self.currentlosses = [[math.inf for _ in range(0, self.n)] for _ in range(self.batchsize)]
        self.completedidx = [False for _ in range(self.batchsize)]
        self.completed = 0

    def searchstep(self):
        self.iter += 1
        bestlosses = np.min(self.currentlosses, axis=1)
        print("\rIteration "+str(self.iter)+" attacked "+str(self.completed)+" of "+str(self.batchsize)+" Avg loss: "+str(np.average(bestlosses)), end='')

        # Perform restarts if enabled
        if self.restart > 0 and self.iter % self.restart == 0:
            self.currentpoints = [[self.gen_random_point() for _ in range(0, self.n)] for _ in range(self.batchsize)]
            self.currentlosses = [[math.inf for _ in range(0, self.n)] for _ in range(self.batchsize)]

        # Perform evolution and check whether we get a better result
        for speccount in range(self.n):
            newpoints = []
            idxs = []
            # Compute one specimen for all samples in the batch
            for idx in range(self.batchsize):
                if self.completedidx[idx]:
                    continue

                # Get population and current information relative to the sample at idx
                # currentpop = self.currentpoints[idx, :]
                # samplelosses = self.currentlosses[idx, :]

                # Generate new specimen for sample idx
                newpoint = []

                # Select random idxs from the current points to generate parents
                parentsidxs = random.sample([i for i in range(self.n) if i != speccount], 3)

                parents = [self.currentpoints[idx][i] for i in parentsidxs]

                # Computes mutations following DE
                R = random.randrange(self.paramcount)
                for i in range(self.paramcount):
                    uniform = random.uniform(0.0, 1.0)
                    if i == R or uniform < self.cr:
                        newpoint.append(parents[0][i] + self.f * (parents[1][i] - parents[2][i]))
                    else:
                        newpoint.append(self.currentpoints[idx][speccount][i])

                # Clamps result to feasible domain
                count = 0
                for elem in self.searchspace:
                    newpoint[count] = np.clip(newpoint[count], elem[0], elem[1])
                    count += 1

                # Intersects difference vector with hyper rectangle to obtain valid solution if mutated is invalid
                # maxidx = -1
                # maxdiff = -1.0
                # maxdim = 0.0
                # minscale = 1.0
                # count = 0
                # for elem in self.searchspace:
                #     lowoverflow = elem[0] - newpoint[count]
                #     highoverflow = newpoint[count] - elem[1]
                #     if lowoverflow > 0.0:
                #         dim = abs(newpoint[count]-self.currentpoints[idx][speccount][count])
                #         if dim > 0.0:
                #             scale = (dim - lowoverflow) / dim
                #             minscale = min(minscale, scale)
                #     elif highoverflow > 0.0:
                #         dim = abs(newpoint[count]-self.currentpoints[idx][speccount][count])
                #         if dim > 0.0:
                #             scale = (dim - highoverflow) / dim
                #             minscale = min(minscale, scale)
                #     count += 1
                #
                # if minscale < 1.0:
                #     # Rescale vector
                #     for i in range(self.paramcount):
                #         newpoint[i] = self.currentpoints[idx][speccount][i] + (newpoint[i] - self.currentpoints[idx][speccount][i]) * minscale
                #     count = 0
                #     for elem in self.searchspace:
                #         lowoverflow = elem[0] - newpoint[count]
                #         highoverflow = newpoint[count] - elem[1]
                #         if lowoverflow > 0.001 or highoverflow > 0.001:
                #             print("azz")
                #             print(lowoverflow)
                #             print(highoverflow)
                #             print(maxdiff)
                #             exit()
                #         count += 1

                newpoints.append(newpoint)
                idxs.append(idx)

            # Send the new points to the adversary (exception raised when maxquery reached)
            try:
                newlosses, newfooled = self.adversary.attacktarget(newpoints, idxs)
            except adversary.EndExecution:
                return

            count = 0
            for idx in idxs:
                if self.currentlosses[idx][speccount] > newlosses[count]:
                    self.currentlosses[idx][speccount] = newlosses[count]
                    self.currentpoints[idx][speccount] = newpoints[count]
                if newfooled[count]:
                    self.completedidx[idx] = True
                    self.completed += 1
                count += 1


# CMA-ES optimizer
class CMAES(Optimizer):
    def __init__(self, searchspace: list, n=10):
        super().__init__(searchspace, 1)  # Batch size greater than 1 not yet implemented
        self.n = n      # Population size
        self.initstep()

        # Set scaling factors
        self.baseoffset = np.array([elem[0] for elem in searchspace])
        self.scalefactor = np.array([elem[1]-elem[0] for elem in searchspace])

    def initstep(self):
        self.currentpoints = np.ones([self.n, len(self.searchspace)])*0.5
        self.es = cma.CMAEvolutionStrategy(self.currentpoints[0, :], 0.5,
                                           {'bounds': [0.0, 1.0], 'verbose': -2})

    def searchstep(self):
        cmapoints = self.es.ask()
        # Rescale from 0-1 to searchspace range
        self.currentpoints = [np.array(point)*self.scalefactor + self.baseoffset for point in cmapoints]

        try:
            losses = self.adversary.attacktarget(self.currentpoints)
        except adversary.EndExecution:
            return
        self.es.tell(cmapoints, losses)


# NGOpt adaptable optimizer using nevergrad ask and tell interface
class NGO(Optimizer):
    def __init__(self, searchspace: list, batchsize: int, budget: int):
        super().__init__(searchspace, batchsize)  # Batch size greater than 1 not yet implemented
        self.budget = budget
        self.initstep()
        self.suggestedpoints = None

        # Set scaling factors
        global SOLUTION_CUBE_RADIUS
        SOLUTION_CUBE_RADIUS = 3.0
        self.baseoffset = np.array([(elem[1]+elem[0])/2.0 for elem in searchspace])
        self.scalefactor = np.array([(elem[1]-elem[0])/(2.0*SOLUTION_CUBE_RADIUS) for elem in searchspace])

        self.nginstr = None
        self.ngoptimizers = None
        self.candidates = None

        self.iter = 0

    def initstep(self):
        self.iter = 0
        # nevergrad requires a new optimizer for each search
        # attention, the optimizer returns results in -pi/2, pi/2
        self.nginstr = ng.p.Instrumentation(ng.p.Array(shape=(self.paramcount,)), y=ng.p.Scalar())
        # Force solution in [-SOLUTION_CUBE_RADIUS SOLUTION_CUBE_RADIUS]
        self.nginstr.register_cheap_constraint(self.isincube)
        self.ngoptimizers = [ng.optimizers.NGOpt(parametrization=self.nginstr, num_workers=10, budget=self.budget) for _ in range(self.batchsize)]

        self.suggestedpoints = np.zeros([self.batchsize, self.paramcount])
        self.completedidx = [False for _ in range(self.batchsize)]
        self.candidates = [None for _ in range(self.batchsize)]
        self.completed = 0

    def searchstep(self):
        self.iter += 1
        print("\rIteration "+str(self.iter)+" attacked "+str(self.completed)+" of "+str(self.batchsize), end='')

        idxs = []
        count = 0
        # Compile the list of new points for those that still have to be attacked
        for idx in range(self.batchsize):
            if self.completedidx[idx]:
                continue
            idxs.append(idx)
            # asks the solution in [-SOLUTION_CUBE_RADIUS, SOLUTION_CUBE_RADIUS]
            # then rescales it to the problem's search space
            self.candidates[idx] = self.ngoptimizers[idx].ask()
            self.suggestedpoints[count, :] = np.array(self.candidates[idx].args)[0, :]
            self.suggestedpoints[count, :] = self.suggestedpoints[count, :] * self.scalefactor + self.baseoffset
            count += 1

        # Send the new points to the adversary (exception raised when maxquery reached)
        try:
            newlosses, newfooled = self.adversary.attacktarget(self.suggestedpoints[0:count, :], idxs)

        except adversary.EndExecution:
            return

        count = 0
        for idx in idxs:
            if newfooled[count]:
                self.completedidx[idx] = True
                self.completed += 1
            self.ngoptimizers[idx].tell(self.candidates[idx], newlosses[count])
            count += 1


# NGOpt adaptable optimizer using nevergrad ask and tell interface
class CutNGO(Optimizer):
    def __init__(self, searchspace: list, batchsize: int, budget: int):
        super().__init__(searchspace, batchsize)  # Batch size greater than 1 not yet implemented
        self.budget = budget
        self.initstep()
        self.suggestedpoints = None

        # Set scaling factors
        global SOLUTION_CUBE_RADIUS
        SOLUTION_CUBE_RADIUS = 3.0
        self.baseoffset = np.array([(elem[1]+elem[0])/2.0 for elem in searchspace])
        self.scalefactor = np.array([(elem[1]-elem[0])/(2.0*SOLUTION_CUBE_RADIUS) for elem in searchspace])

        self.nginstr = None
        self.ngoptimizers = None
        self.candidates = None

        self.iter = 0

    def initstep(self):
        self.iter = 0
        # nevergrad requires a new optimizer for each search
        # attention, the optimizer returns results in -pi/2, pi/2
        self.nginstr = ng.p.Instrumentation(ng.p.Array(shape=(self.paramcount,)), y=ng.p.Scalar())
        # Force solution in [-SOLUTION_CUBE_RADIUS SOLUTION_CUBE_RADIUS]
        self.nginstr.register_cheap_constraint(self.isincube)
        self.ngoptimizers = [ng.optimizers.NGOpt(parametrization=len(self.searchspace), num_workers=10, budget=self.budget) for _ in range(self.batchsize)]

        self.suggestedpoints = np.zeros([self.batchsize, self.paramcount])
        self.completedidx = [False for _ in range(self.batchsize)]
        self.candidates = [None for _ in range(self.batchsize)]
        self.completed = 0

    def searchstep(self):
        self.iter += 1
        print("\rIteration "+str(self.iter)+" attacked "+str(self.completed)+" of "+str(self.batchsize), end='')

        idxs = []
        count = 0
        # Compile the list of new points for those that still have to be attacked
        for idx in range(self.batchsize):
            if self.completedidx[idx]:
                continue
            idxs.append(idx)
            # asks the solution in [-SOLUTION_CUBE_RADIUS, SOLUTION_CUBE_RADIUS]
            # then rescales it to the problem's search space
            self.candidates[idx] = self.ngoptimizers[idx].ask()
            self.suggestedpoints[count, :] = np.array(self.candidates[idx].args)[0, :]

            self.suggestedpoints[count, :] = self.suggestedpoints[count, :] * self.scalefactor + self.baseoffset
            count += 1
        oldpoints = self.suggestedpoints

        # Send the new points to the adversary (exception raised when maxquery reached)
        try:
            self.suggestedpoints[0:count, :] = self.adversary.get_attackmodel_valid_params(self.suggestedpoints[0:count, :], idxs)

            newlosses, newfooled = self.adversary.attacktarget(self.suggestedpoints[0:count, :], idxs)
            self.suggestedpoints[0:count, :] = (self.suggestedpoints[0:count, :] - self.baseoffset)/self.scalefactor

        except adversary.EndExecution:
            return

        count = 0
        for idx in idxs:
            if newfooled[count]:
                self.completedidx[idx] = True
                self.completed += 1

            # if np.any(oldpoints[count, :]-self.suggestedpoints[count, :]):
            #     self.ngoptimizers[idx].tell(oldpoints[count], 10)

            # self.ngoptimizers[idx].tell(self.candidates[idx], newlosses[count])
            # self.ngoptimizers[idx].suggest(self.suggestedpoints[count, :])
            # candidate = self.ngoptimizers[idx].ask()
            # self.ngoptimizers[idx].tell(candidate, newlosses[count])

            #Non modified candidate
            self.ngoptimizers[idx].tell(self.candidates[idx], newlosses[count])

            # candidate = self.ngoptimizers[idx].parametrization.spawn_child(new_value=((list(self.suggestedpoints[idx, :])),))
            count += 1


# Particle swarm optimization using nevergrad ask and tell interface
class PSO(Optimizer):
    def __init__(self, searchspace: list, batchsize: int):
        super().__init__(searchspace, batchsize)  # Batch size greater than 1 not yet implemented
        self.initstep()
        self.suggestedpoints = None

        # Set scaling factors
        global SOLUTION_CUBE_RADIUS
        SOLUTION_CUBE_RADIUS = 3.0
        self.baseoffset = np.array([(elem[1]+elem[0])/2.0 for elem in searchspace])
        self.scalefactor = np.array([(elem[1]-elem[0])/(2.0*SOLUTION_CUBE_RADIUS) for elem in searchspace])

        self.nginstr = None
        self.psoptimizers = None
        self.candidates = None

        self.iter = 0

    def initstep(self):
        self.iter = 0
        # nevergrad requires a new optimizer for each search
        # attention, the optimizer returns results in -pi/2, pi/2
        self.nginstr = ng.p.Instrumentation(ng.p.Array(shape=(self.paramcount,)), y=ng.p.Scalar())
        # Force solution in [-SOLUTION_CUBE_RADIUS SOLUTION_CUBE_RADIUS]
        self.nginstr.register_cheap_constraint(self.isincube)
        self.psoptimizers = [ng.optimizers.RealSpacePSO(parametrization=self.nginstr, num_workers=10) for _ in range(self.batchsize)]

        self.suggestedpoints = np.zeros([self.batchsize, self.paramcount])
        self.completedidx = [False for _ in range(self.batchsize)]
        self.candidates = [None for _ in range(self.batchsize)]
        self.completed = 0

    def searchstep(self):
        self.iter += 1
        print("\rIteration "+str(self.iter)+" attacked "+str(self.completed)+" of "+str(self.batchsize), end='')

        idxs = []
        count = 0
        # Compile the list of new points for those that still have to be attacked
        for idx in range(self.batchsize):
            if self.completedidx[idx]:
                continue
            idxs.append(idx)
            # asks the solution in [-SOLUTION_CUBE_RADIUS, SOLUTION_CUBE_RADIUS]
            # then rescales it to the problem's search space
            self.candidates[idx] = self.psoptimizers[idx].ask()
            self.suggestedpoints[count, :] = np.array(self.candidates[idx].args)[0, :]
            self.suggestedpoints[count, :] = self.suggestedpoints[count, :] * self.scalefactor + self.baseoffset
            count += 1

        # Send the new points to the adversary (exception raised when maxquery reached)
        try:
            newlosses, newfooled = self.adversary.attacktarget(self.suggestedpoints[0:count, :], idxs)

        except adversary.EndExecution:
            return

        count = 0
        for idx in idxs:
            if newfooled[count]:
                self.completedidx[idx] = True
                self.completed += 1
            self.psoptimizers[idx].tell(self.candidates[idx], newlosses[count])
            count += 1


# Particle swarm optimization using nevergrad ask and tell interface
class CutPSO(Optimizer):
    def __init__(self, searchspace: list, batchsize: int):
        super().__init__(searchspace, batchsize)  # Batch size greater than 1 not yet implemented
        self.initstep()
        self.suggestedpoints = None

        # Set scaling factors
        global SOLUTION_CUBE_RADIUS
        SOLUTION_CUBE_RADIUS = 3.0
        self.baseoffset = np.array([(elem[1] + elem[0]) / 2.0 for elem in searchspace])
        self.scalefactor = np.array([(elem[1] - elem[0]) / (2.0 * SOLUTION_CUBE_RADIUS) for elem in searchspace])

        self.nginstr = None
        self.psoptimizers = None
        self.candidates = None

        self.iter = 0

    def initstep(self):
        self.iter = 0
        # nevergrad requires a new optimizer for each search
        # attention, the optimizer returns results in -pi/2, pi/2
        self.nginstr = ng.p.Instrumentation(ng.p.Array(shape=(self.paramcount,)), y=ng.p.Scalar())
        # Force solution in [-SOLUTION_CUBE_RADIUS SOLUTION_CUBE_RADIUS]
        self.nginstr.register_cheap_constraint(self.isincube)
        self.psoptimizers = [ng.optimizers.RealSpacePSO(parametrization=len(self.searchspace), num_workers=1) for _ in
                             range(self.batchsize)]

        self.suggestedpoints = np.zeros([self.batchsize, self.paramcount])
        self.completedidx = [False for _ in range(self.batchsize)]
        self.candidates = [None for _ in range(self.batchsize)]
        self.completed = 0

    def searchstep(self):
        self.iter += 1
        print("\rIteration " + str(self.iter) + " attacked " + str(self.completed) + " of " + str(self.batchsize),
              end='')

        idxs = []
        count = 0
        # Compile the list of new points for those that still have to be attacked
        for idx in range(self.batchsize):
            if self.completedidx[idx]:
                continue
            idxs.append(idx)
            # asks the solution in [-SOLUTION_CUBE_RADIUS, SOLUTION_CUBE_RADIUS]
            # then rescales it to the problem's search space
            self.candidates[idx] = self.psoptimizers[idx].ask()
            self.suggestedpoints[count, :] = np.array(self.candidates[idx].args)[0, :]
            self.suggestedpoints[count, :] = self.suggestedpoints[count, :] * self.scalefactor + self.baseoffset
            count += 1

        # Send the new points to the adversary (exception raised when maxquery reached)
        try:
            self.suggestedpoints[0:count, :] = self.adversary.get_attackmodel_valid_params(
                self.suggestedpoints[0:count, :], idxs)

            newlosses, newfooled = self.adversary.attacktarget(self.suggestedpoints[0:count, :], idxs)
            self.suggestedpoints[0:count, :] = (self.suggestedpoints[0:count, :] - self.baseoffset) / self.scalefactor

        except adversary.EndExecution:
            return

        count = 0
        for idx in idxs:
            if newfooled[count]:
                self.completedidx[idx] = True
                self.completed += 1
            # self.psoptimizers[idx].tell(self.candidates[idx], 1)
            self.psoptimizers[idx].suggest(self.suggestedpoints[count, :])
            candidate = self.psoptimizers[idx].ask()
            # candidate = self.ngoptimizers[idx].parametrization.spawn_child(new_value=((list(self.suggestedpoints[idx, :])),))
            self.psoptimizers[idx].tell(candidate, newlosses[count])
            count += 1

# This optimizer is only to be used with NonParametricPatchAttack
# It is needed because the original Patch-RS implementation is non-parametric.
# It only supports binary colours as per the original implementation
class NonParametricPatchSearch(Optimizer):
    def __init__(self, searchspace: list, patchsize:int, imsize:int, batchsize: int, schedule_iter, schedule_factor, max_iter, alpha_init=0.4, h_init=0.75, m=4):
        super().__init__(searchspace, batchsize)
        self.patchsize = patchsize
        self.imsize = imsize
        self.alpha_init = alpha_init
        self.max_iter = max_iter
        self.h_init = h_init
        self.m = m
        self.currentpoints = None
        self.currentpatches = None
        self.currentlocations = None
        self.currentlosses = math.inf
        self.currentpatches = None
        self.currentlocations = None
        self.initstep()
        self.schedule_iter = schedule_iter
        self.schedule_length = len(self.schedule_iter)
        self.schedule_factor = schedule_factor
        self.iter = 0

    @staticmethod
    def patch_to_searchspace(location, patch):
        size = patch.shape[0]
        return location + [patch[i, j, k] for j in range(size) for i in range(size) for k in range(3)]

    def gen_random_patch(self):
        # Initialize black image
        res = np.zeros([self.patchsize, self.patchsize, 3])

        # Superimpose 1000 random squares
        for i in range(1000):
            size = generator.integers(1, math.ceil(self.patchsize ** .5))  # Following Sparse-RS use sqrt(s) (line 252)
            posx = generator.integers(0, self.patchsize - size + 1)
            posy = generator.integers(0, self.patchsize - size + 1)
            r = generator.integers(0, 2)
            g = generator.integers(0, 2)
            b = generator.integers(0, 2)
            res[posx:posx+size, posy:posy+size, :] = [r, g, b]

        # Image.fromarray((res*255.0).astype(np.uint8)).show()
        # exit()
        return res

    def gen_random_location(self):
        posx = generator.integers(0, self.imsize - self.patchsize)
        posy = generator.integers(0, self.imsize - self.patchsize)

        return [posx, posy]

    def initstep(self):
        self.currentpoints = None
        self.currentpatches = None
        self.currentlocations = None
        self.currentlosses = np.ones([self.batchsize])*np.inf
        self.completedidx = [False for _ in range(self.batchsize)]
        self.completed = 0
        self.iter = 0

    def searchstep(self):
        self.iter += 1
        print("\rIteration "+str(self.iter)+" attacked "+str(self.completed)+" of "+str(self.batchsize)+" Avg loss: "+str(np.average(self.currentlosses)), end='')
        # The first step is different as it does not require interpolation
        if self.currentpoints is None:
            self.currentpatches = [self.gen_random_patch() for _ in range(self.batchsize)]
            self.currentlocations = [self.gen_random_location() for _ in range(self.batchsize)]
            self.currentpoints = [self.patch_to_searchspace(self.currentlocations[i], self.currentpatches[i])
                                  for i in range(self.batchsize)]
            try:
                # Get initial losses from all points, also, if some samples are fooled at the first iteration, save
                # them in self.completedidx. Then initialize self.completed by scanning completedidx.
                self.currentlosses, self.completedidx = self.adversary.attacktarget(self.currentpoints, range(self.batchsize))
                for i in range(self.batchsize):
                    if self.completedidx[i]:
                        self.completed += 1
            except adversary.EndExecution:
                pass
            finally:
                return

        # From the second step onwards, we perform search and check whether we get a better result.
        # Find at what point we are in the schedule
        schedule_idx = self.schedule_length
        for i in range(self.schedule_length):
            if self.schedule_iter[i] > self.iter:
                schedule_idx = i - 1
                break

        # Compute alpha and h from schedule and
        h = (float(self.max_iter - self.iter)/float(self.max_iter))*self.h_init
        shift = math.ceil(h*float(self.imsize))
        s = self.schedule_factor[schedule_idx]
        print(" Schedule status:" + str(schedule_idx) + " Upd_size:" + str(s) + " " + " Shift_size:" + str(shift) + " " + "#" * 10, end='')

        # For samples that were not yet fooled, construct the new points
        newpatches = copy.deepcopy(self.currentpatches)
        newlocations = copy.deepcopy(self.currentlocations)

        # Change location once every m iterations, else add a square
        idxs = []
        count = 0
        if self.iter % self.m == 0 and shift > 0:
            for idx in range(self.batchsize):
                if self.completedidx[idx]:
                    continue

                # Computes a new position for the patch
                deltaposx = int(generator.integers(-shift, shift))
                deltaposy = int(generator.integers(-shift, shift))

                newlocations[idx] = [np.clip(self.currentlocations[idx][0]+deltaposx, 0, self.imsize - self.patchsize - 1),
                                     np.clip(self.currentlocations[idx][1]+deltaposy, 0, self.imsize - self.patchsize - 1)]

                idxs.append(idx)
                count += 1
        else:
            for idx in range(self.batchsize):
                if self.completedidx[idx]:
                    continue

                # Computes a new square to overlay
                updatesize = s

                posx = int(generator.integers(0, self.patchsize - updatesize + 1))
                posy = int(generator.integers(0, self.patchsize - updatesize + 1))

                if updatesize == 1:
                    # Perform single channel mutation for single pixels after iteration 6000
                    if self.iter > 6000:
                        color = self.currentpatches[idx][posx, posy]
                        changechannel = generator.integers(0, 2)
                        color[changechannel] = (color[changechannel] - 1) * -1
                        newpatches[idx][posx, posy, :] = color
                    # Before iteration 6000 just make sure that color is different
                    else:
                        color = self.currentpatches[idx][posx, posy]
                        while True:
                            r = generator.integers(0, 2)
                            g = generator.integers(0, 2)
                            b = generator.integers(0, 2)
                            if r != color[0] or g != color[1] or b != color[2]:
                                break
                        newpatches[idx][posx, posy, :] = color
                else:
                    r = generator.integers(0, 2)
                    g = generator.integers(0, 2)
                    b = generator.integers(0, 2)
                    newpatches[idx][posx:posx+updatesize, posy:posy+updatesize, :] = [r, g, b]

                idxs.append(idx)
                count += 1

        newpoints = [self.patch_to_searchspace(newlocations[i], newpatches[i])
                     for i in idxs]

        # Send the new points to the adversary (exception raised when maxquery reached)
        try:
            newlosses, newfooled = self.adversary.attacktarget(newpoints, idxs)
        except adversary.EndExecution:
            return

        count = 0
        for idx in idxs:
            if self.currentlosses[idx] > newlosses[count]:
                self.currentlosses[idx] = newlosses[count]
                self.currentpatches[idx] = newpatches[count]
                self.currentlocations[idx] = newlocations[count]
                self.currentpoints[idx] = newpoints[count]
            if newfooled[count]:
                self.completedidx[idx] = True
                self.completed += 1
            count += 1


# This is the SparseRS version of the PSO optimizer, to be used only when using eval_original.
# It provides an ask and tell interface and does not ask the adversary.attacktarget
# Moreover, it performs searchsteps based on the idxs_to_fool
# TODO create an ask and tell optimizer class and move isincube and init and isvalidsolution
class SparseRSPSO():
    def __init__(self, searchspace: list, batchsize: int):
        # Search space is a list of 2-tuples that indicate min and max values for every dimension in the space
        self.searchspace = searchspace
        self.paramcount = len(self.searchspace)
        self.adversary = None
        self.batchsize = batchsize
        self.completedidx = [False for _ in range(batchsize)]
        self.completed = 0
        self.initstep()
        self.suggestedpoints = None

        # Set scaling factors
        global SOLUTION_CUBE_RADIUS
        SOLUTION_CUBE_RADIUS = 3.0
        self.baseoffset = np.array([(elem[1]+elem[0])/2.0 for elem in searchspace])
        self.scalefactor = np.array([(elem[1]-elem[0])/(2.0*SOLUTION_CUBE_RADIUS) for elem in searchspace])

        self.nginstr = None
        self.ngoptimizers = None
        self.candidates = None

        self.iter = 0

    # Checks whether all the parameters in the solution are within the search space bounds.
    # This function is used for nevergrad optimizers which require a callable to check it.
    def isvalidsolution(self, inst: ng.p.Instrumentation) -> bool:
        point = inst[0][0]

        count = 0
        for elem in self.searchspace:
            if elem[0] > point[count] > elem[1]:
                continue
            return False
        return True

    # Checks whether all the parameters in the solution are within the search space bounds.
    # This function is used for nevergrad optimizers which require a callable to check it.
    def isincube(self, inst: ng.p.Instrumentation) -> bool:
        global SOLUTION_CUBE_RADIUS
        point = inst[0][0]

        count = 0
        for _ in range(self.paramcount):
            if -1.0*SOLUTION_CUBE_RADIUS < point[count] < 1.0*SOLUTION_CUBE_RADIUS:
                continue
            return False
        return True

    def initstep(self):
        self.iter = 0
        # nevergrad requires a new optimizer for each search
        # attention, the optimizer returns results in -pi/2, pi/2
        self.nginstr = ng.p.Instrumentation(ng.p.Array(shape=(self.paramcount,)), y=ng.p.Scalar())
        # Force solution in [-SOLUTION_CUBE_RADIUS SOLUTION_CUBE_RADIUS]
        self.nginstr.register_cheap_constraint(self.isincube)
        self.ngoptimizers = [ng.optimizers.RealSpacePSO(parametrization=self.nginstr, num_workers=10) for _ in range(self.batchsize)]

        self.suggestedpoints = np.zeros([self.batchsize, self.paramcount])
        self.completedidx = [False for _ in range(self.batchsize)]
        self.candidates = [None for _ in range(self.batchsize)]
        self.completed = 0

    def ask(self, idxs_to_fool: list):
        self.iter += 1

        # Compile the list of new points for those that still have to be attacked
        count = 0
        for idx in idxs_to_fool:
            # asks the solution in [-SOLUTION_CUBE_RADIUS, SOLUTION_CUBE_RADIUS]
            # then rescales it to the problem's search space
            self.candidates[idx] = self.ngoptimizers[idx].ask()
            self.suggestedpoints[count, :] = np.array(self.candidates[idx].args)[0, :]
            self.suggestedpoints[count, :] = self.suggestedpoints[count, :] * self.scalefactor + self.baseoffset
            count += 1

        return self.suggestedpoints[0:count, :]

    def tell(self, idxs_to_fool: list, losses):
        losses = losses.cpu().numpy()
        count = 0
        for idx in idxs_to_fool:
            self.ngoptimizers[idx].tell(self.candidates[idx], losses[count])
            count += 1
