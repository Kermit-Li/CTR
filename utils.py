import os
import torch
import random
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


class ImageDS(Dataset):

    def __init__(self, df, transform=None, train_val="train"):
        self.df = df.copy().reset_index(drop=True)
        self.data_transform = transform
        self.train_val = train_val

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row.file
        x = row.x
        y = row.y
        z = row.z
        rx = row.rx
        ry = row.ry
        rz = row.rz
        a = torch.tensor([x, y], dtype=torch.float32)
        b = torch.tensor(z, dtype=torch.float32)
        c = torch.tensor([rx, ry, rz], dtype=torch.float32)
        if self.data_transform is not None:
            img = Image.open(img_path)
            img = self.data_transform[self.train_val](img)
            return img, a, b, c
        else:
            img = Image.open(img_path)
        return img, a, b, c


# 数据规范化
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        #             transforms.RandomCrop(299),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
}


def training(model, train_dl, val_dl, num_epochs, device):
    # criterion, Optimizer and Scheduler
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(
                                                        len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='cos')

    train_losses = []
    test_losses = []

    begin = datetime.now()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_dl):
            inputs, la, lb, lc = data[0].to(device), data[1].to(
                device), data[2].to(device), data[3].to(device)

            optimizer.zero_grad()
            a, b, c = model(inputs)
            loss = criterion(a, la)+criterion(b, lb)+criterion(c, lc)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        train_loss = running_loss / num_batches
        train_losses.append(train_loss)

        test_loss = valing(model, val_dl, device)
        test_losses.append(test_loss)
        print(
            f'Epoch: {epoch}, Loss: {train_loss:.4f}, Val loss: {test_loss:.4f}, Time: {datetime.now() - begin}')

    print('Finished Training')
    return train_losses, test_losses


def valing(model, val_dl, device):
    criterion = torch.nn.SmoothL1Loss()
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data in val_dl:
            inputs, la, lb, lc = data[0].to(device), data[1].to(
                device), data[2].to(device), data[3].to(device)

            a, b, c = model(inputs)
            loss = criterion(a, la)+criterion(b, lb)+criterion(c, lc)
            running_loss += loss.item()

    num_batches = len(val_dl)
    avg_loss = running_loss / num_batches

    return avg_loss


def result_plot(train_losses, test_losses):
    epochs = len(train_losses)

    plt.plot(np.linspace(1, epochs, epochs), train_losses, label="train_loss")
    plt.plot(np.linspace(1, epochs, epochs), test_losses, label="test_loss")
    plt.title("LOSS Plot")
    plt.legend()
    plt.savefig("loss.jpg")
