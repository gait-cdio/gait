from model import CifarResNeXt
import torch
import argparse

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Positional arguments
parser.add_argument('--data_path', type=str, default="data", help='Root for the Cifar dataset.')
parser.add_argument('--dataset', type=str, default=['cifar100'], choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--test_bs', type=int, default=10)
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Architecture
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# i/o
parser.add_argument('--log', type=str, default='./', help='Log folder.')
args = parser.parse_args()

nlabels = 1
net = CifarResNeXt(args.cardinality, args.depth, nlabels, args.widen_factor)
loaded_state_dict = torch.load(args.load)
temp = {}
for key, val in list(loaded_state_dict.items()):
    # parsing keys for ignoring 'module.' in keys
    temp[key[7:]] = val
loaded_state_dict = temp
net.load_state_dict(loaded_state_dict)