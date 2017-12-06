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


has_graphics = bool(os.environ.get('DISPLAY'))

select_points_interactively = True

if select_points_interactively:
    fig, ax = plt.subplots()
    ax.imshow(image)

    click_recorder = ClickRecorder(fig)

    plt.show()
    clicks = click_recorder.clicks
else:
    # Poor person's caching. TODO(rolf): replace with real caching
    clicks = [(438.947004608295, 879.72488479262677), (584.93778801843314, 836.95990783410139),
              (127.79493087557603, 898.89539170506919), (42.264976958525324, 789.77096774193546)]

groundtruth_output = np.zeros(image.shape[0:2])

for click in clicks:
    width, height, _ = image.shape
    x, y = range(width), range(height)
    xx, yy = np.meshgrid(y, x)

    sigma = 100.0
    groundtruth_output += np.exp(- ((xx - click[0]) ** 2 + (yy - click[1]) ** 2) / (2 * sigma ** 2))

window_shape = (32, 32, 3)
training_images = skimage.util.shape.view_as_windows(image, window_shape, step=1)

#training_samples = np.zeros(100, 32, 32, 3)

reshaped_training_images = training_images.reshape(-1, 32, 32, 3)
groundtruth_windows = skimage.util.shape.view_as_windows(groundtruth_output, (32, 32), step=1)
reshaped_groundtruth_output = groundtruth_windows.reshape(-1, 32, 32)[:, 15, 15]

#for index, _ in enumerate(training_samples):


dataset = torch.utils.data.TensorDataset(torch.ByteTensor(reshaped_training_images),
                                         torch.Tensor(reshaped_groundtruth_output))

vgg = models.vgg11(pretrained=True)

for param in vgg.parameters():
    param.requires_grad = False

vgg.features = torch.nn.Sequential(*[vgg.features[i] for i in range(8)])
vgg.classifier = torch.nn.Linear(16384, 1)

vgg = vgg.cuda()

lossfunction = torch.nn.MSELoss()
optimizer = torch.optim.SGD(vgg.classifier.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

for sample in dataset:
    input, groundtruth = sample

    input = input.float().permute(2, 0, 1).unsqueeze_(0)
    groundtruth = torch.Tensor([groundtruth]).unsqueeze_(0)

    input, groundtruth = Variable(input).cuda(), Variable(groundtruth).cuda()

    optimizer.zero_grad()

    output = vgg(input)
    loss = lossfunction(output, groundtruth)

    print("Loss:", loss.data[0])

    loss.backward()
    optimizer.step()
