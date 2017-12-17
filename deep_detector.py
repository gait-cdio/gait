import cv2
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision

import deep_train


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


def load_groundtruth(video_filename, patch_size=64, cached=False):
    groundtruth_filename = get_groundtruth_filename(video_filename)

    input_cache_filename = groundtruth_filename + '_input_cache.npy'
    output_cache_filename = groundtruth_filename + '_output_cache.npy'

    if cached and os.path.isfile(input_cache_filename):
        print("Found cached groundtruth data!")
        return np.load(input_cache_filename), np.load(output_cache_filename)

    position_groundtruth = np.load(groundtruth_filename)

    cap = cv2.VideoCapture(video_filename)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    left_toe, left_heel = 0, 1
    right_toe, right_heel = 2, 3
    interesting_body_parts = [left_heel, right_heel]

    half_patch_size = patch_size // 2

    input_patches = np.zeros(shape=(num_frames * 2 * len(interesting_body_parts), patch_size, patch_size, 3))
    output_patches = np.zeros(shape=(num_frames * 2 * len(interesting_body_parts), patch_size, patch_size))

    sample_index = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_index in range(num_frames):
        if np.all(np.isnan(position_groundtruth[frame_index, interesting_body_parts, 0])):
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index) # This one takes time and is not needed, I think
        ret, image = cap.read()
        assert ret

        height, width, _ = image.shape
        x, y = range(width), range(height)
        xx, yy = np.meshgrid(x, y)
        groundtruth_output = np.zeros(shape=(height, width))

        sigma = 2

        interesting_positions = position_groundtruth[frame_index, interesting_body_parts][~np.isnan(position_groundtruth[frame_index, interesting_body_parts, 0])]

        def sample_image(image, x, y):
            return image[y - half_patch_size:y + half_patch_size,
                   x - half_patch_size:x + half_patch_size]

        def clamp(n, smallest, largest):
            return max(smallest, min(n, largest))

        interesting_positions = list(
            filter(lambda a:
                       half_patch_size < a[0] < width - half_patch_size
                       and
                       half_patch_size < a[1] < height - half_patch_size,
                   interesting_positions))
        for pos in interesting_positions:
            x, y = pos
            groundtruth_output += np.exp(- ((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

        for pos in interesting_positions:
            x, y = int(pos[0]), int(pos[1])
            input_patches[sample_index] = sample_image(image, x, y)
            output_patches[sample_index] = sample_image(groundtruth_output, x, y)
            sample_index += 1

            x2, y2 = x, y - 1.5 * patch_size
            x2 += patch_size * np.random.randn()
            y2 += patch_size * np.random.randn()
            x2 = int(clamp(x2, half_patch_size, width - half_patch_size))
            y2 = int(clamp(y2, half_patch_size, height - half_patch_size))
            input_patches[sample_index] = sample_image(image, x2, y2)
            output_patches[sample_index] = sample_image(groundtruth_output, x2, y2)
            sample_index += 1

    input_patches = input_patches[:sample_index]
    output_patches = output_patches[:sample_index]
    np.save(input_cache_filename, input_patches)
    np.save(output_cache_filename, output_patches)
    return input_patches, output_patches


def run_network(network, x):
    x = network.features(x)
    x = network.classifier(x)
    return x


def classify_pixels(trained_model, image):
    out = run_network(trained_model,
                      Variable(torch.from_numpy(image)
                               .permute(2, 0, 1)
                               .unsqueeze(0))
                      .float()
                      .cuda())

    output_patch = out.data.cpu().numpy()[0, 0]
    lilpatch = output_patch[3:-3, 3:-3]

    return lilpatch


def main():
    patch_size = 220
    half_patch_size = patch_size // 2

    train = False

    if train:
        input_data = {}
        output_data = {}
        for person in ['john', 'kevin', 'rolf']:
            video_filename = "input-images/{}_markerless/{}_markerless_%04d.jpg".format(person, person)
            print("Loading data for {} ...".format(person))
            input_data[person], output_data[person] = load_groundtruth(video_filename, patch_size=patch_size, cached=False)
            print("Loaded data for {}.".format(person))

        training_examples = np.concatenate(list(input_data.values()))
        training_example_outputs = np.concatenate(list(output_data.values()))

        permutation_indices = np.random.permutation(len(training_example_outputs))
        shuffled_training_examples = training_examples[permutation_indices]
        shuffled_training_example_outputs = training_example_outputs[permutation_indices]

        foot_dataset = separate_dataset_into_train_val(shuffled_training_examples, shuffled_training_example_outputs)

        data_loaders = {x: torch.utils.data.DataLoader(foot_dataset[x], batch_size=50,
                                                       shuffle=True, num_workers=4)
                        for x in ['train', 'val']}
    else:
        person = 'kevin'
        video_filename = "input-images/{}_markerless/{}_markerless_%04d.jpg".format(person, person)

    if train:
        vgg = torchvision.models.vgg19(pretrained=True)
        vgg.features = torch.nn.Sequential(*[vgg.features[i] for i in range(4)])

        for params in vgg.parameters():
            params.require_grad = False

        vgg.classifier = torch.nn.Conv2d(64, 1, kernel_size=3, padding=1)

        if torch.cuda.is_available():
            vgg = vgg.cuda()

        criterion = torch.nn.MSELoss()
        optimizer_conv = optim.SGD(vgg.classifier.parameters(), lr=1e-7, momentum=0.5)
        num_epochs = 100
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv)

        trained_model = deep_train.train_model(vgg, criterion, optimizer_conv, exp_lr_scheduler, data_loaders, run_network,
                                               num_epochs=num_epochs)

        torch.save(trained_model.state_dict(), "cnnetworks/deeptracker_state_dict")
        torch.save(trained_model, "cnnetworks/deeptracker")
    else:
        trained_model = torch.load("cnnetworks/deeptracker")

    cap = cv2.VideoCapture(video_filename)
    cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)
    tracking = True
    if tracking:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter('output-videos/deeptracker' + person +  '.avi', fourcc, fps, (width, height))
        frameList = range(20,150)
    else:
        frameList = [31,41,51]

    prevMax = [patch_size, 870]
    for frame in frameList:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, image = cap.read()

        assert ret

        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[0:2]

        if tracking:
            fig, ax = plt.subplots()
            click_recorder = ClickRecorder(fig)
            plt.close()
            click_recorder.clicks = [1]
        else:
            fig, ax = plt.subplots()
            ax.imshow(display_image)
            ax.set_title('Please select regions to evaluate')
            click_recorder = ClickRecorder(fig)
            plt.show()


        maxes = []
        for coords in enumerate(click_recorder.clicks):


            if tracking:
                x, y = int(prevMax[0]), int(prevMax[1])
            else:
                x, y = int(coords[0]), int(coords[1])

            col_low, col_high = max(0, x - half_patch_size), min(width, x + half_patch_size)
            row_low, row_high = max(0, y - half_patch_size), min(height, y + half_patch_size)

            input_patch = image[row_low:row_high, col_low:col_high]
            out = run_network(trained_model,
                              Variable(torch.from_numpy(input_patch)
                                       .permute(2, 0, 1)
                                       .unsqueeze(0))
                              .float()
                              .cuda())

            output_patch = out.data.cpu().numpy()[0, 0]
            lilpatch = output_patch[3:-3, 3:-3]

            temp = np.zeros(lilpatch.shape)
            for tempx in range(lilpatch.shape[0]):
                for tempy in range(lilpatch.shape[1]):
                    temp[tempx, tempy] = np.sqrt((tempx - lilpatch.shape[0] / 2) ** 2 + (tempy - lilpatch.shape[1]/ 2) ** 2)
            distanceMask = np.cos(np.multiply(temp, np.pi / (np.max(temp) * 2)))
            lilpatch_masked = np.multiply(lilpatch, distanceMask)

            maxrow, maxcol = np.unravel_index(lilpatch_masked.argmax(), lilpatch_masked.shape)
            maxrow += 3
            maxcol += 3
            maxes.append((maxrow + y - half_patch_size, maxcol + x - half_patch_size))

            prevMax = [maxcol + x - half_patch_size, maxrow + y - half_patch_size]

            clamped = np.fmax(np.zeros(lilpatch_masked.shape), lilpatch_masked)
            grayscale_patch = np.array(np.stack((clamped,) * 3, axis=-1) * 255 / np.max(clamped), dtype=np.uint8)

            alpha = 1
            display_image[row_low+3:row_high-3, col_low+3:col_high-3] = alpha * grayscale_patch + (1 - alpha) * input_patch[3:-3, 3:-3]

        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

        if not tracking:
            for y, x in maxes:
                plt.scatter(x, y, c='blue', marker='x', s=24)
        else:
            for y, x in maxes:
                cv2.circle(display_image, (x,y), 5, (0,255,0), thickness=-1)
        cv2.waitKey(1)
        cv2.imshow("Tracking", display_image)
        if tracking:
            writer.write(display_image)
    writer.release()
    cap.release()

if __name__ == '__main__':
    main()
