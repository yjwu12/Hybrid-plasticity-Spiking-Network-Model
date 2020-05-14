# -*- coding: utf-8 -*-
"""
Spiking EWC model

@author: yjwu

Python 3.5.2

"""

from __future__ import print_function
import sys
import torchvision
import torchvision.transforms as transforms
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time
from spiking_ewc_model import*


data_path =  r'./' # Download the MNIST dataset and modify the data path
saving_name = 'mnist_mlp_ewc'


train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


n = 0
lambda_loss = 1e-3

def train():

    criterion = nn.MSELoss()
    total_acc_record = []

    for numbers in range(exp_num):
        setup_seed(numbers)
        print('set the number:', numbers)
        snn = SNN_Model()
        snn.to(device)
        optimizer = torch.optim.Adam(snn.parameters(), lr=5e-4)

        acc_record = []

        hebb1, hebb2, hebb3 = snn.produce_hebb()
        total_avg = []
        running_loss = 0.
        snn.train()
        start_time = time.time()

        total = 0.
        correct = 0.

        W = {}
        p_old = {}

        for n, p in snn.named_parameters():
            if p.requires_grad:
                print(n)
                n = n.replace('.', '__')
                snn.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
                snn.register_buffer('{}_SI_omega'.format(n), 0 * p.data.clone())


        for train_task_index in range(task_num):
            ss = np.arange(28 * 28)
            if train_task_index > 0:
                np.random.seed(train_task_index)
                np.random.shuffle(ss)

            for n, p in snn.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

            for epoch in range(num_task_epochs):
                train_correct = 0.
                train_total = 0.
                snn.epoch = epoch

                for i, (inputs, labels) in enumerate(train_loader):

                    optimizer.zero_grad()
                    numpy_data = inputs.view(batch_size, -1).data.cpu().numpy()
                    images = torch.from_numpy(numpy_data[:, ss])

                    images, labels = images.float().to(device), labels.to(device)
                    outputs, spikes, mems, hebb1, hebb2, hebb3, eta1, eta2 = snn(input=images, hebb1=hebb1, hebb2=hebb2,hebb3 = hebb3,
                                                                          wins=time_window, label=labels)
                    targets_ = torch.zeros(batch_size, 10, device=device).scatter_(1, labels.view(-1, 1), 1)

                    # surrogate_loss = snn.surrogate_loss()

                    surrogate_loss = snn.ewc_loss()

                    loss = criterion(outputs, targets_) + snn.ewc_lambda *surrogate_loss


                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum()


                    for n, p in snn.named_parameters():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                W[n].add_(-p.grad * (p.detach() - p_old[n]))
                            p_old[n] = p.detach().clone()

                print('\n\n Task [{:d}/{:d}]: Epoch [{:d}/{:d}], Iter [{:d}/{:d}] Loss: {:.5f} Surrogate loss: {:.5f}'
                      .format(train_task_index + 1, task_num, epoch + 1, num_task_epochs, i + 1,
                              len(train_dataset) // batch_size,
                              loss.data[0], surrogate_loss ))
                print('Epoch %0.2d  Train Accuracy of the model on the 60000 Shuffled_mnist images: %0.2f %%' % (epoch,
                        100 * float(train_correct) / float(train_total)))

            # snn.update_omega(W, 0.1)
            snn.estimate_fisher(train_loader, input_parameter=(hebb1, hebb2, hebb3), permutted_paramer=ss,
                                allowed_classes=None)


            # Test the Model
            correct_all = []
            for task_index in range(train_task_index + 1):
                ss = np.arange(28 * 28)
                if task_index > 0:
                    np.random.seed(task_index)
                    np.random.shuffle(ss)
                correct = 0.
                total = 0.
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.float().to(device), labels.to(device)

                        numpy_data = images.view(batch_size, -1).data.cpu().numpy()
                        images = torch.from_numpy(numpy_data[:, ss]).cuda()

                        outputs, spikes, mems, _, _, _, _,_ = snn(input=images, hebb1=hebb1, hebb2=hebb2,hebb3=hebb3,
                                                                wins=time_window, label=labels)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    correct_all.append((100 * float(correct) / float(total)))

            if train_task_index>=0:
                print("Task Number : %.2f, Average Test Accuracy on All Tasks: {0:%.2f} "% (train_task_index, sum(correct_all) / len(correct_all)))
                total_avg.append(sum(correct_all) / len(correct_all))

        total_acc_record.append(total_avg)

def main():
    train()

if __name__ == '__main__':
    main()







