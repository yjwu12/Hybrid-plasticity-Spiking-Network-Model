# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 09:46:25 2018

@author: yjwu

Python 3.5.2

"""

from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from mnist_model import*


data_path =  './' # input your data path
names = 'mnist_model'
lambdas = 0.

train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
criterion = nn.MSELoss()


exp_num = 1
for exp_idx in range(exp_num):
    snn = SNN_Model()
    snn.to(device)
    param_base, param_local = snn.parameter_split()
    optim_base = torch.optim.Adam(param_base, lr = 5e-4)
    optim_local = torch.optim.Adam(param_local, lr = 5e-4)

    acc_record = []
    hebb1 , hebb2 = snn.produce_hebb()
    for epoch in range(num_epochs):
        running_loss = 0.
        snn.train()
        start_time = time.time()

        total = 0.
        correct = 0.

        train_step_record = list([])
        test_step_record = list([])
        train_opts_record = list([])
        test_opts_record = list({})

        for i, (images, targets) in enumerate(train_loader):
            snn.zero_grad()
            images = images.float().to(device)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)

            for i_update in range(num_updates):
                optim_base.zero_grad()
                outputs, spikes, hebb1, hebb2, step = snn(input=images, hebb1=hebb1, hebb2=hebb2,wins=time_window)
                loss = criterion(outputs.cpu(), labels_)
                loss.backward()
                optim_base.step()
                if i_update < (num_updates - 1):optim_local.zero_grad()

            optim_local.step()
            optim_local.zero_grad()
            running_loss += loss.item()

            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())

            if i % (60000/batch_size/4) == 0 and i >1:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Accuray: %.5f'%(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,
                                                                                 running_loss, float(correct)/float(total)*100 ))

        correct = 0.
        total = 0.
        running_loss = 0.
        snn.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                outputs, _, _, _, _ = snn(input=inputs, hebb1=hebb1, hebb2=hebb2, wins=time_window)
                targets_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))

                correct += float(predicted.eq(targets).sum().item())

            acc = 100. * float(correct) / float(total)

        print('Iters:', epoch,'\n\n\n')
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))

        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)
        if epoch > 40:
            if acc > best_acc  :
                best_acc = acc
                print(acc)
                print('Saving..')







