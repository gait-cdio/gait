import cv2
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision

import test_dl_utils


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


def separate_dataset_into_train_val(inputs, outputs, ratio=0.6):
    num_inputs = inputs.shape[0]
    num_outputs = outputs.shape[0]
    assert num_inputs == num_outputs

    splitting_index = int(num_inputs * ratio)

    train_inputs = inputs[:splitting_index]
    val_inputs = inputs[splitting_index:]

    train_outputs = outputs[:splitting_index]
    val_outputs = outputs[splitting_index:]

    return {'train': TensorDataset(torch.from_numpy(train_inputs).float(), torch.from_numpy(train_outputs).float()),
            'val': TensorDataset(torch.from_numpy(val_inputs).float(), torch.from_numpy(val_outputs).float())}


def get_groundtruth_filename(video_filename):
    filename_without_folder = os.path.split(video_filename)[1]
    if '%04d' in video_filename:
        video_name = filename_without_folder.split('_%04d')[0]
        groundtruth_filename = 'annotations/' + video_name + '-positions.npy'
    else:
        video_name = os.path.splitext(filename_without_folder)[0]
        groundtruth_filename = 'annotations/' + video_name + '-positions.npy'
    return groundtruth_filename


def load_groundtruth(video_filename):
    groundtruth_filename = get_groundtruth_filename(video_filename)

    position_groundtruth = np.load(groundtruth_filename)

    cap = cv2.VideoCapture(video_filename)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    input_patches = np.zeros(shape=(num_frames * 4, 64, 64, 3))
    output_patches = np.zeros(shape=(num_frames * 4, 64, 64))

    sample_index = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_index in range(num_frames):
        if np.all(np.isnan(position_groundtruth[frame_index, :, 0])):
            continue

        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index) # This one takes time and is not needed, I think
        ret, image = cap.read()
        assert ret

        height, width, _ = image.shape
        x, y = range(width), range(height)
        xx, yy = np.meshgrid(x, y)
        groundtruth_output = np.zeros(shape=(height, width))

        sigma = 10.0
        interesting_positions = position_groundtruth[frame_index][
            np.where(~np.isnan(position_groundtruth[frame_index, :, 0]))]
        interesting_positions = list(filter(lambda a: 40 < a[0] < width - 40, interesting_positions))
        for pos in interesting_positions:
            x, y = pos
            groundtruth_output += np.exp(- ((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

        for pos in interesting_positions:
            x, y = int(pos[0]), int(pos[1])
            input_patches[sample_index] = image[y - 32:y + 32, x - 32:x + 32]
            output_patches[sample_index] = groundtruth_output[y - 32:y + 32, x - 32:x + 32]
            sample_index += 1

    return input_patches, output_patches


video_filename = "input-images/rolf_markerless/rolf_markerless_%04d.jpg"
positive_examples, positive_example_outputs = load_groundtruth(video_filename)

np.random.shuffle(positive_examples)
np.random.shuffle(positive_example_outputs)

foot_dataset = separate_dataset_into_train_val(positive_examples, positive_example_outputs)

data_loaders = {x: torch.utils.data.DataLoader(foot_dataset[x], batch_size=50,
                                               shuffle=True, num_workers=1)
                for x in ['train', 'val']}

vgg = torchvision.models.vgg19(pretrained=True)
vgg.features = torch.nn.Sequential(*[vgg.features[i] for i in range(4)])

for params in vgg.parameters():
    params.require_grad = False

vgg.classifier = torch.nn.Conv2d(64, 1, kernel_size=3, padding=1)

if torch.cuda.is_available():
    vgg = vgg.cuda()

criterion = torch.nn.MSELoss()
optimizer_conv = optim.SGD(vgg.classifier.parameters(), lr=1e-6)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1, gamma=0.1, last_epoch=-1)

trained_model = test_dl_utils.train_model(vgg, criterion, optimizer_conv, exp_lr_scheduler, data_loaders,
                                          num_epochs=50)

cap = cv2.VideoCapture(video_filename)

for frame in [31, 61, 91]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, image = cap.read()

    assert ret

    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[0:2]

    fig, ax = plt.subplots()
    ax.imshow(display_image)
    ax.set_title('Please select regions to evaluate')
    click_recorder = ClickRecorder(fig)
    plt.show()

    for index, coords in enumerate(click_recorder.clicks):
        x, y = int(coords[0]), int(coords[1])
        col_low, col_high = max(0, x - 32), min(width, x + 32)
        row_low, row_high = max(0, y - 32), min(height, y + 32)

        input_patch = image[row_low:row_high, col_low:col_high]
        out = trained_model(Variable(torch.from_numpy(input_patch)
                                     .permute(2, 0, 1)
                                     .unsqueeze(0))
                            .float()
                            .cuda())

        output_patch = out.data.cpu().numpy()[0, 0]
        grayscale_patch = np.stack((output_patch,) * 3, axis=-1)
        alpha = 1
        display_image[row_low:row_high, col_low:col_high] = alpha * grayscale_patch + (1 - alpha) * input_patch

    plt.imshow(display_image)
    plt.show()
