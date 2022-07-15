from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from tsrdmodel.data import data_transforms

BATCH_SIZE = 64
REPORT_INTERVAL = 10

def train(epoch):
    model.train()
    correct = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        max_index = output.max(dim=1)[1]
        correct += (max_index == target).sum()
        training_loss += loss
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss per example: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), loss.data.item() / (BATCH_SIZE * REPORT_INTERVAL),
                loss.data.item()))
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


if __name__ == "__main__":
    if torch.cuda.is_available():
        use_gpu = True
        print("Using GPU")
    else:
        use_gpu = False
        print("Using CPU")

    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
    Tensor = FloatTensor

    # Apply data transformations on the training images to augment dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('N:/PycharmProjects/scratchthat/datasets/tsrd_mask/train',
                             transform=data_transforms), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=use_gpu)

    from tsrdmodel.model import Net
    model = Net()

    if use_gpu:
        model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    model_file = None
    for epoch in range(1, 201):
        train(epoch)
        model_file = 'model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print(
            '\nSaved model to ' + model_file + '. Run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')

    torch.save(model.state_dict(), model_file)
