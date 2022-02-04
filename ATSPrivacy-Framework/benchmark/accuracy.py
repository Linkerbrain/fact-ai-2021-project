import os
import sys

sys.path.insert(0, '../')
import torch
import torch.nn as nn
import math

import numpy as np

seed = 23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random

random.seed(seed)

import inversefed
import argparse
from benchmark.comm import create_model, preprocess

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')
# parser.add_argument('--optim', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
parser.add_argument('--resume', default=0, type=int, help='rlabel')

opt = parser.parse_args()
num_images = 1

# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative');
defs.epochs = opt.epochs

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']


# config = create_config(opt)


def create_checkpoint_dir():
    return 'checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list,
                                                                             opt.rlabel)


def accuracy(predictions, targets):
    avg_accuracy = torch.sum(torch.argmax(predictions, axis=-1) == targets) / math.prod(targets.shape)
    return avg_accuracy


# loop for every epoch (training + evaluation)
def evaluation(model, validloader, epochs, batches):
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0
    # set model to evaluating (testing)
    model.eval()
    val_losses = 0
    accuracies = []
    with torch.no_grad():
        for i, data in enumerate(validloader):
            X, y = data[0].to(device), data[1].to(device)
            outputs = model(X)  # this get's the prediction from the network

            val_losses += loss_function(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1]  # get class from network's prediction
            accuracies.append(accuracy(outputs, y).cpu())
    return np.average(accuracies)
    # losses.append(total_loss/batches) # for plotting learning curve


def main():
    global trained_model
    print(opt)
    loss_fn, trainloader, validloader = preprocess(opt, defs, valid=True)
    model = create_model(opt)
    model.to(**setup)
    epoch_list = np.append(np.arange(0, 200, 10), 199)
    accuracies = []
    if opt.epochs == 0:
        trained_model = False

    if trained_model:
        checkpoint_dir = create_checkpoint_dir()
        if 'normal' in checkpoint_dir:
            checkpoint_dir = checkpoint_dir.replace('normal', 'crop')
        filename = os.path.join(checkpoint_dir, str(defs.epochs) + '.pth')

        if not os.path.exists(filename):
            filename = os.path.join(checkpoint_dir, str(defs.epochs - 1) + '.pth')
        for epoch in epoch_list:
            root = './checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}/'.format(opt.data, opt.arch, opt.mode,
                                                                                        opt.aug_list, opt.rlabel)
            filename = root + str(epoch) + '.pth'
            print(filename)
            # assert os.path.exists(filename)
            model.load_state_dict(torch.load(filename))
            epochs = 200
            batches = 64
            accuracies.append(evaluation(model, validloader, epochs, batches))
        accuracy_root = './checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}/'.format(opt.data, opt.arch,
                                                                                             opt.mode, opt.aug_list,
                                                                                             opt.rlabel)
        np.save('{}/val_accuracies.npy'.format(accuracy_root), accuracies)
        print(accuracies)

    if opt.rlabel:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = False


if __name__ == '__main__':
    main()
