import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import transforms
import time
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import inspect

import matplotlib.pyplot as plt
import numpy as np
import inversefed

class MnistResNet(nn.Module):
    def __init__(self, in_channels=1):
        super(MnistResNet, self).__init__()

        # Load a pretrained resnet model from torchvision.models in Pytorch
        self.model, _ = inversefed.construct_model('ResNet20', num_classes=10, num_channels=1)

        # Change the input layer to take Grayscale image, instead of RGB images.
        # Hence in_channels is set as 1 or 3 respectively
        # original definition of the first layer on the ResNet class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        print(self.model.conv1)
        # Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)


my_resnet = MnistResNet()
#
input = torch.randn((16,1,244,244))
output = my_resnet(input)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def get_data_loaders(train_batch_size, val_batch_size):
    mnist = torchvision.datasets.MNIST(download=True , train=True, root=".").train_data.float()

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((mnist.mean()/255,), (mnist.std()/255,))])

    train_set = torchvision.datasets.MNIST(download=True, root=".", transform=data_transform, train=True)
    val_set = torchvision.datasets.MNIST(download=True, root=".", transform=data_transform, train=False)

    tiny_train_set = torch.utils.data.Subset(train_set, range(0, int(0.1*len(train_set))))
    tiny_val_set = torch.utils.data.Subset(val_set, range(0, int(0.1*len(val_set))))

    train_loader = DataLoader(tiny_train_set,
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(tiny_val_set,
                            batch_size=val_batch_size, shuffle=False)


    return train_loader, val_loader

# model:
model = MnistResNet().to(device)

# params you need to specify:
epochs = 50
batch_size = 64

# Dataloaders
train_loader, val_loader = get_data_loaders(batch_size, batch_size)

# loss function and optimiyer
loss_function = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

# optimizer, I've used Adadelta, as it wokrs well without any magic numbers
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) # Using Karpathy's learning rate constant

start_ts = time.time()

losses = []
batches = len(train_loader)
val_batches = len(val_loader)

def accuracy(predictions, targets):
    avg_accuracy = torch.sum(torch.argmax(predictions, axis=-1) == targets) / math.prod(targets.shape)
    return avg_accuracy
# loop for every epoch (training + evaluation)
for epoch in range(epochs):
    total_loss = 0

    # progress bar (works in Jupyter notebook too!)
    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

    # ----------------- TRAINING  --------------------
    # set model to training
    model.train()

    for i, data in progress:
        X, y = data[0].to(device), data[1].to(device)

        # training step for single batch
        model.zero_grad()
        outputs = model(X)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()

        # getting training quality data
        current_loss = loss.item()
        total_loss += current_loss

        # updating progress bar
        progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))

    # releasing unceseccary memory in GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------- VALIDATION  -----------------
    val_losses = 0
    accuracy = [], [], [], []

    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X, y = data[0].to(device), data[1].to(device)

            outputs = model(X) # this get's the prediction from the network

            val_losses += loss_function(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction

    print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    losses.append(total_loss/batches) # for plotting learning curve
print(f"Training time: {time.time()-start_ts}s")
torch.save(model.state_dict(), "./Models/MnistResNet20-4.pth")
