import cv2
import numpy as np
import torch
import matplotlib.pyplot as py
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
import test_dl_utils


def selectHeel(filename, frame):
    cap = cv2.VideoCapture(filename)
    cap.set(1, frame)
    ret, img = cap.read()
    cv2.namedWindow("Bild")
    if ret:

        roi = cv2.selectROI("Bild", img)
        cv2.destroyAllWindows()
        cv2.normalize(img, img, dtype=cv2.CV_32F)
        x, y, w, h = roi
        height, width, _ = img.shape
        mask = np.zeros([height, width])
        mask[y:y + h, x:x + w] = 1

        heelArray = []
        for i in range(x - 15, x + w - 15):
            for k in range(y - 15, y + h - 15):
                heelArray.append(img[k:k + 32, i:i + 32])

        falseArray = []
        for i in range(x - 32, x + w):
            falseArray.append(img[y - 32:y, i:i + 32])
            falseArray.append(img[y + h:y + h + 32, i:i + 32])
        for i in range(y - 32, y + h):
            falseArray.append(img[i:i + 32, x - 32:x])
            falseArray.append(img[i:i + 32, x + w:x + w + 32])
        return mask, np.array(heelArray), np.array(falseArray)
    else:
        return False


def separateDatasets(trueArray, falseArray, ratio=0.6):
    num_true_images = trueArray.shape[0]
    num_false_images = falseArray.shape[0]
    true_splitting_index = int(num_true_images * ratio)
    false_splitting_index = int(num_false_images * ratio)
    # TODO(rolf): handle corner case with too few images (splitting index becomes < 0)

    train_images = np.concatenate((trueArray[:true_splitting_index], falseArray[:false_splitting_index]), axis=0)
    val_images   = np.concatenate((trueArray[true_splitting_index:], falseArray[false_splitting_index:]), axis=0)

    train_labels = np.zeros(len(train_images), dtype="int")
    val_labels = np.zeros(len(val_images), dtype="int")

    train_labels[:true_splitting_index] = 1
    train_labels[true_splitting_index:] = 0
    val_labels[:(num_true_images - true_splitting_index)] = 1
    val_labels[(num_true_images - true_splitting_index):] = 0

    trainTensor = torch.from_numpy(train_images)
    trainInts = torch.from_numpy(train_labels)
    valTensor = torch.from_numpy(val_images)
    valInts = torch.from_numpy(val_labels)

    return {'train': TensorDataset(trainTensor.float(), trainInts),
            'val': TensorDataset(valTensor.float(), valInts)}


mask, iA, fA = selectHeel("input-videos/4farger.mp4", 120)

np.random.shuffle(iA)
np.random.shuffle(fA)

footDataset = separateDatasets(iA, fA)

dataloaders = {x: torch.utils.data.DataLoader(footDataset[x], batch_size=50,
                                              shuffle=True, num_workers=1)
               for x in ['train', 'val']}

vgg19 = torchvision.models.vgg19(pretrained=True)
vgg19.features = torch.nn.Sequential(*[vgg19.features[i] for i in range(8)])

for params in vgg19.parameters():
    params.require_grad = False
fc1 = torch.nn.Linear(32768, 32)
fc2 = torch.nn.Linear(32,2)
vgg19.classifier = torch.nn.Sequential(fc1,fc2)

if torch.cuda.is_available():
    vgg19 = vgg19.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(vgg19.classifier.parameters(), lr=0.00001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1, gamma=0.1, last_epoch=-1)

trained_model = test_dl_utils.train_model(vgg19, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, num_epochs=20)


_, new_iA, new_fA = selectHeel("input-videos/4farger.mp4", 119)


for im in new_iA:

    out = trained_model(Variable(torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)).float().cuda())
    _, pred = torch.max(out.data, 1)
    print(pred)
    cv2.imshow('test bild', im)
    cv2.waitKey(0)

'''for phase in ['train', 'val']:
    for data in dataloaders[phase]:
        image, label = data
        print(image.shape)
        print(label.shape)'''
