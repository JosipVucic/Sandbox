import time

import torch
import torch.nn as nn
import torchvision
from digits.neural.model import GACNN
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from tqdm import tqdm

"""
You can run this file to train and save a model to trained_models/default.pth.
There is a model already there so you don't actually have to run this.
Actual training was performed using a Jupyter notebook with this code on Google Colab.
See "DigitsModelTrain.ipynb" for notebook outputs.
"""

def train(genotype=None, batch_size=32, epoch_limit=1000):
    """
    Loads MNIST dataset. Trains the model defined by genotype on MNIST for a number of epochs.
    Training will stop after epoch limit is reached or after the validation accuracy stops rising.

    :param genotype: Genotype for GACNN model
    :param batch_size: Batch size for data loader
    :param epoch_limit: Maximum number of epochs to train model
    :return: Trained model.
    """

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainset, validationset = random_split(trainset, [0.8, 0.2])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    num_classes = 10
    in_chans = 1

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = GACNN(genotype=genotype, num_classes=num_classes, in_chans=in_chans)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model.to(device)

    acc_best, epoch_best, epoch, acc, loss = (0, 0, 0, 0, 0)

    t0 = time.time()
    while (acc >= acc_best or epoch - epoch_best < 5) and epoch < epoch_limit:

        for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        acc = evaluate_model(model, validationloader)

        if acc > acc_best:
            acc_best, epoch_best = (acc, epoch)

        epoch += 1

        print(f"Epoch {epoch} - loss = {loss} | validation accuracy = {acc}")

    t1 = time.time()

    train_acc = evaluate_model(model, trainloader)
    test_acc = evaluate_model(model, testloader)
    print(f"Epoch {epoch} - train accuracy = {train_acc} | test accuracy = {test_acc}")
    print(f"Training time: {t1 - t0}")

    return model


def evaluate_model(model, testloader):
    """
    Evaluates model on given dataloader.

    :param model: PyTorch model to evaluate
    :param testloader: DataLoader object with evaluation dataset
    :return: Accuracy as percentage
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    trained_model = train()
    torch.save(trained_model.state_dict(), 'trained_models/default.pth')

    # pip.main(['install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])
