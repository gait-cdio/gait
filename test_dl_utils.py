import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os


def train_model(model, criterion, optimizer, scheduler, data_loaders, run_network, num_epochs=25):
    since = time.time()

    best_model_weights = model.state_dict()
    best_loss = np.inf

    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for data in data_loaders[phase]:
                # get the inputs
                inputs, truths = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.permute(0, 3, 1, 2).cuda())
                    truths = Variable(truths.cuda())
                else:
                    inputs = inputs.permute(0, 3, 1, 2)
                    inputs, truths = Variable(inputs), Variable(truths)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = run_network(model, inputs)
                loss = criterion(outputs, truths)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                else:
                    scheduler.step(loss.data[0])

                # statistics
                running_loss += loss.data[0] / len(inputs)

            epoch_loss = running_loss

            if phase == 'val':
                val_losses[epoch] = epoch_loss
            else:
                train_losses[epoch] = epoch_loss

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_weights)
    plt.semilogy(train_losses, label='Training loss')
    plt.semilogy(val_losses, label='Validation loss')
    plt.legend()
    plt.show()
    return model



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class ImageArrDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return {'image': self.images[item], 'label': self.labels[item]}
