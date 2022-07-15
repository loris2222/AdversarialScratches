from target import Target
from attackmodel import AttackModel
from abc import ABC, abstractmethod
import warnings
import numpy as np
from keras.losses import binary_crossentropy
from utils import slice_by_index
import torch
import torch.nn.functional as F
import copy

class EndExecution(Exception):
    pass


class AdversaryLoss(ABC):
    def __init__(self):
        pass

    @abstractmethod
    # Computes the fitness of the attack based on the output of the model and the original sample. The loss depends on
    # whether the attack is targeted/untargeted and whether the model's output is a label or a vector.
    # The function must return the loss value and a bool indicating whether the model was fooled given its output.
    def __call__(self, original_output, adversarial_output) -> (float, bool):
        pass


class Adversary(object):
    def __init__(self, target: Target, optimizer, loss: AdversaryLoss, attackmodel: AttackModel):
        self.target = target
        self.optimizer = optimizer
        self.attackmodel = attackmodel
        self.searchspacesize = len(attackmodel.searchspace)
        self.loss = loss
        self.successidx = None
        self.successqueries = None
        self.successcount = 0
        self.batch_size = 0
        self.successparams = None
        self.queries = 0
        self.maxqueries = 0
        self.original_output = None
        self.sample_batch = None

    # Loops the optimizer to iteratively obtain adversarial samples. The samples are directly sent to the model through
    # attacktarget by the optimizer. If the attack is successful, self.success will be the adversarial sample.
    def performattack(self, sample_batch, maxqueries: int) -> (np.ndarray, int, np.ndarray):
        self.successidx = np.zeros([sample_batch.shape[0]])  # Indicates which samples are fooled
        self.successqueries = np.zeros([sample_batch.shape[0]]).astype(int)  # Indicates which samples are fooled
        self.successcount = 0
        self.successparams = np.zeros([sample_batch.shape[0], len(self.attackmodel.searchspace)])
        self.queries = 0
        self.maxqueries = maxqueries
        self.optimizer.initstep()
        self.sample_batch = sample_batch
        self.batch_size = sample_batch.shape[0]
        self.original_output = self.target.predict(sample_batch[:, :, :, 0:3])

        while self.successcount < self.batch_size and self.queries < self.maxqueries:
            self.optimizer.searchstep()
        return self.successidx, self.successqueries, self.successparams

    # Queries the target and returns the output. This function is called by the optimizer.
    # attacktarget also checks the loss condition for the target to be fooled, and returns which of the samples in the
    # batch fooled the model with a bool list.
    # In case query limit is reached, the optimizer execution is stopped through an exception.
    # Because we are batching, we need the optimizer to tell us which samples we are attacking, this is idxs purpose.
    def attacktarget(self, param_batch: list, batch_idxs: list):
        # If query limit is reached, stop execution of the optimizer.
        # This will trigger return values from performattack.
        self.queries += 1
        if self.queries > self.maxqueries or self.successcount == self.batch_size:
            raise EndExecution("query limit reached")

        # Modify samples given the parameters using self.attackmodel
        # then predict the model output on the modified samples.
        outputs = self.target.predict(self.attackmodel(self.sample_batch[batch_idxs], param_batch))

        # Compute loss for each sample (its on CPU so we don't need batching).
        # If sample fooled, update info (success count, idx, and params).
        scores = []
        fooled = []
        count = 0
        for idx in batch_idxs:
            cur_score, cur_fooled = self.loss(self.original_output[idx], outputs[count])
            scores.append(cur_score)
            fooled.append(cur_fooled)
            if cur_fooled:
                self.successcount += 1
                self.successidx[idx] = 1
                self.successqueries[idx] = self.queries
                self.successparams[idx, :] = np.array(param_batch[count]).reshape([1, self.searchspacesize])
            count += 1

        # Returns losses results to the optimizer so that it can know which samples were fooled (and remove them from
        # future batches).
        return scores, fooled

    # For CutMaskedBezierLine, it returns the parameter configuration which results in a Bézier fully contained in mask
    def get_attackmodel_valid_params(self, param_batch: list, batch_idxs: list):
        return self.attackmodel.get_valid_params(param_batch, self.sample_batch[batch_idxs])


# Returns the score of the target, threshold is a lower bound.
class TargetedMarginalLoss(AdversaryLoss):
    def __init__(self, targetidx, threshold):
        super().__init__()
        self.targetidx = targetidx
        self.threshold = threshold

    def __call__(self, original_output, adversarial_output) -> (float, bool):
        if original_output[self.targetidx] > self.threshold:
            warnings.warn("Target label is already predicted")
        score = 1 - adversarial_output[self.targetidx]
        fooled = score > self.threshold

        return score, fooled


# Returns the crossentropy between [0, 0, ...., 1, 0, ...., 0] and the model's posterior. where 1 is in the target pos.
# Threshold is computed on the score of the target, same as in TargetedScoreOnlyLoss
class TargetedCrossEntropyLoss(AdversaryLoss):
    def __init__(self, targetidx, threshold):
        super().__init__()
        self.targetidx = targetidx
        self.threshold = threshold

    def __call__(self, original_output, adversarial_output) -> (float, bool):
        target = np.zeros(original_output.shape[0])
        target[self.targetidx] = 1
        score = binary_crossentropy(target, adversarial_output)
        fooled = adversarial_output[self.targetidx] > self.threshold

        return score, fooled


# Returns the posterior of the class that was originally classified. Threshold is an upper bound.
class UntargetedScoreLoss(AdversaryLoss):
    def __init__(self):
        super().__init__()

    def __call__(self, original_output, adversarial_output) -> (float, bool):
        score = adversarial_output[np.argmax(original_output)]
        fooled = (np.argmax(original_output) != np.argmax(adversarial_output))

        return score, fooled


class MarginLoss(AdversaryLoss):
    def __init__(self):
        super().__init__()

    def __call__(self, original_output, adversarial_output) -> (float, bool):
        # Output class of non perturbed sample (possibly the correct label)
        original_max = np.argmax(original_output)
        # Create a copy of the adversarial output and set the correct label posterior to -inf
        adversarial_nomax = adversarial_output.copy()
        adversarial_nomax[original_max] = -np.inf

        score = adversarial_output[original_max] - np.max(adversarial_nomax)
        fooled = score <= -1e-6  # 0.0  # (np.argmax(original_output) != np.argmax(adversarial_output))

        return score, fooled


# Same as MarginLoss but takes torch tensors instead of numpy arrays (it still returns numpy array for ask/tell interf.)
class TorchMarginLoss(AdversaryLoss):
    def __init__(self):
        super().__init__()

    def __call__(self, original_output, adversarial_output) -> (float, bool):
        # Output class of non perturbed sample (possibly the correct label)
        original_max = torch.argmax(original_output)
        # Create a copy of the adversarial output and set the correct label posterior to -inf
        adversarial_nomax = adversarial_output.clone()
        adversarial_nomax[original_max] = -np.inf

        score = (adversarial_output[original_max] - torch.max(adversarial_nomax)).cpu().detach().numpy().tolist()
        fooled = score <= -1e-6  # 0.0  # (np.argmax(original_output) != np.argmax(adversarial_output))

        return score, fooled


# Should be the same as MarginLoss above but double checked with Sparse-RS implementation
class SparseRSLoss(AdversaryLoss):
    def __init__(self, num_classes):
        super().__init__()
        self.NUM_CLASSES = num_classes

    def __call__(self, original_output, adversarial_output) -> (float, bool):
        logits = torch.from_numpy(adversarial_output.reshape([1, self.NUM_CLASSES]))
        y = np.array(np.argmax(original_output)).reshape([1])
        y_corr = logits[0, y].clone()
        logits[0, y] = -float('inf')
        y_others = logits.max(dim=-1)[0]

        return y_corr - y_others, (y_corr - y_others) <= -1e-6


# To be used when testing the image captioning model from azure. It only considers the model's confidence
# The model must be the azure one as this loss requires model outputs to be of type ImageDescription
class ConfidenceLoss(AdversaryLoss):
    def __init__(self, maximize=False):
        self.maximize = maximize
        super().__init__()

    def __call__(self, original_output, adversarial_output) -> (float, bool):
        original_confidence = original_output.captions[0].confidence
        adversarial_confidence = adversarial_output.captions[0].confidence
        if self.maximize:
            return original_confidence - adversarial_confidence, False
        else:
            return adversarial_confidence - original_confidence, False
