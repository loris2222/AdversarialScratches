from abc import ABC, abstractmethod
import numpy as np
from keras.applications import ResNet50
import tensorflow as tf
import os
from PIL import Image

import torch
import torch.nn as nn

from torchvision import models as torch_models

from gtsrbmodel.model import Net
from gtsrbmodel.data import data_transforms as gtsrbttransforms
from tsrdmodel.data import data_transforms as tsrdtransforms

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO

physical_devices = tf.config.list_physical_devices('GPU')

for h in physical_devices:
    tf.config.experimental.set_memory_growth(h, True)


class Target(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    # Returns the model output, this method is instantiated differently depending on the attack type, so as to give out
    # only the relevant information. The target might return a score or a decision.
    def predict(self, sample):
        pass


class ResNet50ScoreBased(Target):
    def __init__(self):
        super().__init__(ResNet50(weights='imagenet'))

    def predict(self, sample):
        assert sample.dtype == np.uint8
        proces_sample = tf.keras.applications.imagenet_utils.preprocess_input(sample, data_format=None, mode='caffe')
        if len(sample.shape) == 3:
            raise ValueError("Input should not be flattened")
            # return np.array(self.model(proces_sample.reshape([1, 224, 224, 3]))).flatten()
        else:
            samplecount = sample.shape[0]
            return np.array(self.model(proces_sample.reshape([samplecount, 224, 224, 3])))


class PyTorchResNet50(Target):
    def __init__(self):
        model_pt = torch_models.resnet50(pretrained=True)
        # model.eval()
        model = nn.DataParallel(model_pt.cuda())
        model.eval()
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()
        super().__init__(model)

    def predict(self, sample):
        if len(sample.shape) == 3:
            raise ValueError("Input should not be flattened")
            # if sample.shape == [3, 224, 224]:
            #     # sample = sample.transpose([1, 2, 0])
            #     raise ValueError("Possible flattening error, check code")
            # sample = sample.reshape([1, 224, 224, 3])

        normalized = torch.from_numpy(sample.transpose([0, 3, 1, 2]).astype(np.float32)/255.0).cuda()
        out = (normalized - self.mu) / self.sigma
        return self.model(out).cpu().data.numpy()


class ResNet50DecisionBased(Target):
    def __init__(self):
        super().__init__(ResNet50(weights='imagenet'))

    def predict(self, sample):
        return np.argmax(self.model(sample))


class PyTorchGTSRB(Target):
    def __init__(self):
        model_pt = Net()
        # TODO change from hardcoded path to some parametrized version
        state_dict = torch.load(os.path.join(os.getcwd(), "../gtsrbmodel/model_40.pth"))
        model_pt.load_state_dict(state_dict)
        model = nn.DataParallel(model_pt.cuda())
        model.eval()
        self.transforms = [gtsrbttransforms]
        super().__init__(model)

    def predict(self, sample):
        if len(sample.shape) == 3:
            raise ValueError("Input should not be flattened")

        # Resize and offset for dataset statistics (look at gtsrb.data)
        out = torch.zeros([sample.shape[0], 3, 32, 32])
        for i in range(sample.shape[0]):
            im = Image.fromarray(sample[i, :, :, :])
            # TODO if more transforms are added you need a loop here (but then the model is an ensemble)
            out[i, :, :, :] = self.transforms[0](im)

        output = self.model(out)

        return output.cpu().data.numpy()


class PyTorchTSRD(Target):
    def __init__(self):
        model_pt = torch_models.resnet50(pretrained=True)
        num_ftrs = model_pt.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_pt.fc = nn.Linear(num_ftrs, 58)  # 58 classes in dataset

        # TODO change from hardcoded path to some parametrized version
        state_dict = torch.load(os.path.join(os.getcwd(), "./tsrdmodel/resnet_train.pth"))
        model_pt.load_state_dict(state_dict)
        model = nn.DataParallel(model_pt.cuda())
        model.eval()
        self.transforms = [tsrdtransforms]
        super().__init__(model)

    def predict(self, sample):
        if len(sample.shape) == 3:
            raise ValueError("Input should not be flattened")

        # Resize and offset for dataset statistics (look at gtsrb.data)
        out = torch.zeros([sample.shape[0], 3, 224, 224])
        for i in range(sample.shape[0]):
            im = Image.fromarray(sample[i, :, :, :])
            # TODO if more transforms are added you need a loop here (but then the model is an ensemble)
            out[i, :, :, :] = self.transforms[0](im)

        output = self.model(out)

        return output.cpu().data.numpy()


# Same as the other resnet50 but this takes normalized [0..1] images
class PyTorchResNet50Normalized(Target):
    def __init__(self):
        model_pt = torch_models.resnet50(pretrained=True)
        # model.eval()
        model = nn.DataParallel(model_pt.cuda())
        model.eval()
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()
        super().__init__(model)

    def predict(self, x):
        out = (x - self.mu) / self.sigma
        return self.model(out)

    def __call__(self, x):
        return self.predict(x)


# Microsoft Azure captioning
class AzureCaptioning(Target):
    def __init__(self):
        self.imid = 0
        for f in os.listdir("./azure/imagecaptions"):
            os.remove(os.path.join("./azure/imagecaptions", f))
        with open(os.path.join(os.getcwd(), "./azure/key.txt")) as f:
            self.key = f.read()
        with open(os.path.join(os.getcwd(), "./azure/endpoint.txt")) as f:
            self.endpoint = f.read()
        self.computervision_client = ComputerVisionClient(self.endpoint, CognitiveServicesCredentials(self.key))
        super().__init__(None)

    def predict(self, sample):
        res = []
        for i in range(sample.shape[0]):
            # Load image as PIL image
            x = sample[0, :, :, :]
            image = Image.fromarray(x)
            # Get image stream
            out = BytesIO()
            image.save(out, format="BMP")
            out.seek(0)
            # Get description from azure
            description = self.computervision_client.describe_image_in_stream(out)
            res.append(description)
            image.save("./azure/imagecaptions/" + str(description.captions[0].confidence) + "_" + str(self.imid) + "_" + description.captions[0].text + ".bmp", format="BMP")
            self.imid += 1
            return res
