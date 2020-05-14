# -*- coding: utf-8 -*-
"""
Hybrid plasticity model on CIFAR10

@author: yjwu
"""

from __future__ import print_function
import sys


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os, time

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from cifar10_model import *

names = 'cifar_model'

data_path = './data'# input your data path


lambdas = 0.
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

net = SCNN()
# net.apply(init_weights)
net = net.to(device)
criterion = nn.MSELoss()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

param_base, param_local = net.parameter_split()
optim_base = torch.optim.SGD(param_base, lr=learning_rate, momentum=0.9, weight_decay=1e-8)
optim_local = torch.optim.Adam(param_local, lr=5e-4)


def train(epoch, hebb1, hebb2):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # starts = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        starts_ = time.time()
        inputs = inputs.to(device)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)

        for i_update in range(num_update):
            optim_base.zero_grad()
            outputs, step, hebb1, hebb2, eta1, eta2 = net(inputs, hebb1, hebb2)
            loss = criterion(outputs.cpu(), labels_)
            loss.backward()
            optim_base.step()
            if i_update < (num_update - 1): optim_local.zero_grad()

        optim_local.step()
        optim_local.zero_grad()
        train_loss += loss.item()
        _, predicted = outputs.cpu().max(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()
        if batch_idx % 100 == 0:
            #
            elapsed = time.time() - starts
            print(batch_idx, 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print('Time past: ', elapsed, 's', 'Iter number:', epoch)
    # print('b3: %.7f  b4: %.7f' % (torch.sum(net.b3), torch.sum(net.b5)))
    loss_train_record.append(train_loss)
    return hebb1, hebb2


def test(epoch, hebb1, hebb2):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0.
    total = 0.
    sys_opt_record = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)

            outputs, sys_opts, _, _, eta1, eta2 = net(inputs, hebb1, hebb2)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            test_loss += loss.item()
            _, predicted = outputs.cpu().max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()
            # if batch_idx % 50 == 0:
            sys_opt_record.append(sys_opts)
        print(batch_idx, len(testloader), 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        loss_test_record.append(test_loss)

    # Save checkpoint.
    acc = 100. * correct / total

    if best_acc < acc:
        best_acc = acc
        print('Saving..')

    return sys_opt_record, loss_test_record, acc


hebb1, hebb2 = net.produce_hebb()

scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_base, T_max=num_epochs)


total_test_loss = []
total_sys_opts = []
acc_record = []
best_acc = 0.
for epoch in range(num_epochs):
    starts = time.time()
    hebb1, hebb2 = train(epoch, hebb1, hebb2)
    test_loss_record, sys_record, acc = test(epoch, hebb1, hebb2)
    total_test_loss.append(test_loss_record)
    total_sys_opts.append(sys_record)
    acc_record.append(acc)
    elapsed = time.time() - starts
    scheduler.step()
    print(" \n\n\n\n\n\n\n")
    print('Time past: ', elapsed, 's', 'Iter number:', epoch)