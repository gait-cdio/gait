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


def selectHeel(image):
    img = np.copy(image)
    cv2.namedWindow("Bild")
    roi = cv2.selectROI("Bild", img)
    cv2.destroyAllWindows()
    cv2.normalize(img, img, dtype=cv2.CV_32F)
    x, y, w, h = roi
    height, width, _ = img.shape
    mask = np.zeros([height, width])
    mask[y:y + h, x:x + w] = 1

    heelArray = []
    coordArray = []
    for i in range(x - 15, x + w - 15):
        for k in range(y - 15, y + h - 15):
            heelArray.append(img[k:k + 32, i:i + 32])
            coordArray.append((i + 15, k + 15))

    falseArray = []
    for i in range(x - 32, x + w):
        falseArray.append(img[y - 32:y, i:i + 32])
        falseArray.append(img[y + h:y + h + 32, i:i + 32])
    for i in range(y - 32, y + h):
        falseArray.append(img[i:i + 32, x - 32:x])
        falseArray.append(img[i:i + 32, x + w:x + w + 32])
    return np.array(heelArray), np.array(falseArray), coordArray


def separateDatasets(trueArray, falseArray, ratio=0.6):
    num_true_images = trueArray.shape[0]
    num_false_images = falseArray.shape[0]
    true_splitting_index = int(num_true_images * ratio)
    false_splitting_index = int(num_false_images * ratio)
    # TODO(rolf): handle corner case with too few images (splitting index becomes < 0)

    train_images = np.concatenate((trueArray[:true_splitting_index], falseArray[:false_splitting_index]), axis=0)
    val_images = np.concatenate((trueArray[true_splitting_index:], falseArray[false_splitting_index:]), axis=0)

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


cap = cv2.VideoCapture("input-images/rolf_markerless/rolf_markerless_%04d.jpg")
cap.set(1, 30)
ret, image = cap.read()

assert ret

iA, fA, _ = selectHeel(image)

np.random.shuffle(iA)
np.random.shuffle(fA)

footDataset = separateDatasets(iA, fA)

dataloaders = {x: torch.utils.data.DataLoader(footDataset[x], batch_size=50,
                                              shuffle=True, num_workers=1)
               for x in ['train', 'val']}

vgg = torchvision.models.vgg19(pretrained=True)
vgg.features = torch.nn.Sequential(*[vgg.features[i] for i in range(8)])

for params in vgg.parameters():
    params.require_grad = False
fc1 = torch.nn.Linear(32768, 32)
fc2 = torch.nn.Linear(32, 2)
vgg.classifier = torch.nn.Sequential(fc1, fc2)

if torch.cuda.is_available():
    vgg = vgg.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(vgg.classifier.parameters(), lr=0.00001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1, gamma=0.1, last_epoch=-1)

trained_model = test_dl_utils.train_model(vgg, criterion, optimizer_conv, exp_lr_scheduler, dataloaders,
                                          num_epochs=20)

cap.set(1, 31)
ret, image = cap.read()

assert ret

new_iA, new_fA, coords = selectHeel(image)

for index, coord in enumerate(coords):
    out = trained_model(Variable(torch.from_numpy(new_iA[index])
                                 .permute(2, 0, 1)
                                 .unsqueeze(0))
                        .float()
                        .cuda())
    _, prediction = torch.max(out.data, dim=1)

    print('Predicted {}/{}: {}'.format(index + 1, len(coords), bool(prediction[0])))
    if prediction[0]:
        image[coord[1], coord[0]] = (0, 0, 255)

cv2.imshow('Here it is', image)
cv2.waitKey(0)
