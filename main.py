import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from dsets import MNISTDataset
from model import ViT
from einops import rearrange
import time
import torch.nn.functional as F


print("Path of torch: " + torch.__file__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())



def evaluate(model, data_loader, loss_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')



#################
eval_count = 1000

data_train = pd.read_csv('mnist/train.csv')
train_data = data_train.drop('label', axis=1).values
train_mean = train_data.mean()/255.
train_std = train_data.std()/255.   

train_transform = transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean], std=[train_std]),
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean], std=[train_std]),
])

test_transform = val_transform

#train_dataset = MNISTDataset(data_train.iloc[:-eval_count], default_transform)
train_dataset = MNISTDataset(data_train, train_transform) # use this to train the model on the full training set
eval_dataset = MNISTDataset(data_train.iloc[-eval_count:], val_transform)



data_test = pd.read_csv('mnist/test.csv')
test_dataset = MNISTDataset(data_test, test_transform, is_test=True)




torch.manual_seed(42)

BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 1000

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)


model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128)
model = model.to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

train_loss_history, test_loss_history = [], []

N_EPOCHS = 100

# lambda1 = lambda epoch: epoch // 30
# lambda2 = lambda epoch: 0.95 ** epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

start_time = time.time()
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch,'LR:', scheduler.get_last_lr())
    train_epoch(model, optimizer, train_loader, train_loss_history)
    evaluate(model, eval_loader, test_loss_history)
    scheduler.step()
    break

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
