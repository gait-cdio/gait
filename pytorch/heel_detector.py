import numpy as np

import os
import skimage
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models
from PIL import Image

import matplotlib.pyplot as plt

data_dir = '../input-images/john_markerless'
image = np.array(Image.open(os.path.join(data_dir, 'john_markerless_0033.jpg')))


class ClickRecorder:
    def __init__(self, fig):
        self.cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.clicks = []
        self.markers = None

    def onclick(self, event):
        if event.button == 1:
            if event.xdata is not None and event.ydata is not None:
                self.clicks.append((event.xdata, event.ydata))
        elif event.button == 3:
            new_click = event.xdata, event.ydata
            self.clicks = list(
                filter(lambda click: np.linalg.norm((click[0] - new_click[0], click[1] - new_click[1])) > 10,
                       self.clicks))

        if self.markers is not None:
            self.markers.remove()
        self.markers = plt.scatter([click[0] for click in self.clicks],
                                   [click[1] for click in self.clicks], s=20, marker='x', c='red')
        plt.show()


if os.environ.get('DISPLAY') is not None:
	fig, ax = plt.subplots()
	ax.imshow(image)

	click_recorder = ClickRecorder(fig)

	plt.show()
	clicks = click_recorder.clicks
else:
	clicks = [(64, 433)]

groundtruth_output = np.zeros(image.shape[0:2])

for click in clicks:
    width, height, _ = image.shape
    x, y = range(width), range(height)
    xx, yy = np.meshgrid(y, x)

    sigma = 10.0
    groundtruth_output += np.exp(- ((xx - click[0]) ** 2 + (yy - click[1]) ** 2) / (2 * sigma ** 2))

if os.environ.get('DISPLAY') is not None:
	gtfig, gtax = plt.subplots()
	gtax.imshow(groundtruth_output)
	plt.show()

window_shape = (32, 32, 3)
training_images = skimage.util.shape.view_as_windows(image, window_shape)

reshaped_training_images = training_images.reshape(-1, 32, 32, 3)
reshaped_groundtruth_output = groundtruth_output[16:-15,16:-15].reshape(-1)

dataset = torch.utils.data.TensorDataset(torch.ByteTensor(reshaped_training_images), torch.Tensor(reshaped_groundtruth_output))

resnet = models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

num_features = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_features, 1)
resnet.fc.requires_grad = True

resnet = resnet.cuda()

lossfunction = torch.nn.MSELoss()
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

for sample in dataset:
    input, groundtruth = sample
    print(input, groundtruth)
    input, groundtruth = Variable(input).cuda(), Variable(torch.Tensor([groundtruth])).cuda()

    optimizer.zero_grad()

    output = resnet(input)
    loss = lossfunction(output, groundtruth)

    print("Loss:", loss.data[0])

    loss.backward()
    optimizer.step()
