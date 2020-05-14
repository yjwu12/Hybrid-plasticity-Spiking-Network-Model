# -*- coding: utf-8 -*-
"""
Created on Sat Dec 1 2019

@author: Mingkun Xu

Python 3.5.2

"""

from __future__ import print_function
import sys
import torchvision
import torchvision.transforms as transforms
import os
import torch
import time
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# parser.add_argument('--r', type=str, default='1')
parser.add_argument('--p', type=str, default=r'../data/fashion/')
parser.add_argument('--f', type=str, default='../Checkp-fashion')
parser.add_argument('--names', type=str, default='mlp_v1')
parser.add_argument('--model', type=str, default='mask_model_1',choices=['mask_model_0','mask_model_1','mask_model_2'])
opt = parser.parse_args()

if opt.model == 'mask_model_0':
    from model_hp import *
elif opt.model == 'mask_model_1':
    from model_gp import *
# elif opt.model == 'mask_model_2':
#     from mask_model_2 import *

data_path = opt.p
save_path = opt.f #+ '-r{}'.format(opt.r)
names = opt.names
para_path = opt.f + '/' + names + '.pth'
hebb_path = opt.f + '/' + names + 'HEBB.t7'
lambdas = 0.

print('Begin test..')
batch_size = 100

# train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_set = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=5)

criterion = nn.MSELoss()

data = {}
data['crop_i'] = []
data['acc'] = []
data['spike'] = []
data['spike2'] = []

para = torch.load(para_path)
HEBB = torch.load(hebb_path)
snn = SNN_Model()
hebb1 = HEBB['hebb1']
hebb2 = HEBB['hebb2']

snn.load_state_dict(para)
snn.eval()
print(snn)
snn.to(device)

for i in range(15):
    correct = 0.
    total = 0.
    Spike = 0.
    Spike2 = 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs[:, 0, 14 - i:14 + i, 14 - i:14 + i] = 0
            inputs = inputs.to(device)

            outputs, sumspike, sumspike2, _, _, _, _ = snn(input=inputs, hebb1=hebb1, hebb2=hebb2, wins=time_window)

            targets_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), targets_)

            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            Spike += sumspike.detach() / 8.0
            Spike2 += sumspike2.detach() / 8.0

        acc = 100. * float(correct) / float(total)
        Spikem = Spike.mean().cpu().numpy()
        Spikem2 = Spike2.mean().cpu().numpy()

    # print('Model:', names)
    print('{%s}||crop-i: %d || Test Accuracy : %.3f || Spike1:  %.3f || Spike2:  %.3f' % (names, i, acc, Spikem, Spikem2))

    data['crop_i'].append(i)
    data['acc'].append(acc)
    data['spike'].append(Spikem)
    data['spike2'].append(Spikem2)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    torch.save(data, save_path + '/' + names + '_test.t7')

print('Complete..')








