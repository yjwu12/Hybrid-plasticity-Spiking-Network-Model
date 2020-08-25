"""
HP model for multitask learning
"""


from __future__ import print_function
import sys
import torchvision
import torchvision.transforms as transforms
import os
import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import time
from hp_model import*


data_path =  r'./' # Download the MNIST dataset and modify the data path
saving_name = 'hp_model'



train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


def train():
    best_acc = 0.  # best test accuracy
    criterion = nn.MSELoss()
    correct_all = []
    total_best_acc = []
    total_acc_record = []

    for numbers in range(exp_num):
        print('set the number:', numbers)
        setup_seed(numbers)
        total_avg = []


        mask_dict = mask_weight(p=sparsity, task_number=task_num)
        snn = SNN_Model().to(device)
        param_base, param_local = snn.parameter_split()
        optim_base = torch.optim.Adam(param_base, lr=5e-4)
        optim_local = torch.optim.Adam(param_local, lr=5e-5)

        running_loss = 0.
        snn.train()
        hebb1, hebb2, hebb3 = snn.produce_hebb()
        total_random = []


        for train_task_index in range(task_num):

            ss = np.arange(28 * 28)
            if train_task_index > 0:
                np.random.seed(train_task_index)
                np.random.shuffle(ss)
            total_random.append(ss)


            for epoch in range(num_task_epochs):
                train_correct = 0.
                train_total = 0.
                snn.epoch = epoch

                for i, (inputs, labels) in enumerate(train_loader):
                    optim_base.zero_grad()
                    numpy_data = inputs.view(batch_size, -1).data.cpu().numpy()
                    images = torch.from_numpy(numpy_data[:, ss])

                    images, labels = images.float().to(device), labels.to(device)
                    outputs, spikes, mems, hebb1, hebb2, hebb3, _, _ = snn(input=images, hebb1=hebb1, hebb2=hebb2,hebb3 = hebb3,
                                                                          wins=time_window, label=labels, mask_dict = mask_dict[train_task_index])
                    targets_ = torch.zeros(batch_size, 10, device=device).scatter_(1, labels.view(-1, 1), 1)

                    surrogate_loss = 0.

                    loss = criterion(outputs, targets_ )
                    running_loss += loss.item()
                    loss.backward()
                    optim_base.step()

                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum()


                print('\n\n Task [{:d}/{:d}]: Epoch [{:d}/{:d}], Iter [{:d}/{:d}] Loss: {:.5f} Surrogate loss: {:.5f}'
                      .format(train_task_index + 1, task_num, epoch + 1, num_task_epochs, i + 1,
                              len(train_dataset) // batch_size,
                              loss.data[0], surrogate_loss ))
                print('Epoch %0.2d  Train Accuracy of the model on the 60000 Shuffled_mnist images: %0.2f %%' % (epoch,
                        100 * float(train_correct) / float(train_total)))

            if train_task_index < train_meta_epoch:
                # train the meta-parameters
                for epoch in range(1):
                    train_correct = 0.
                    train_total = 0.

                    snn.epoch = epoch
                    optim_local.zero_grad()
                    optim_base.zero_grad()

                    for i, (inputs, labels) in enumerate(train_loader):
                        if train_task_index > 0:
                            random_task_index = np.random.randint(train_task_index)
                        else:
                            random_task_index = 0

                        numpy_data = inputs.view(batch_size, -1).data.cpu().numpy()
                        images = torch.from_numpy(numpy_data[:, total_random[random_task_index]])

                        images, labels = images.float().to(device), labels.to(device)
                        outputs, spikes, mems, _, _, _, _, _ = snn(input=images, hebb1=hebb1, hebb2=hebb2,hebb3 = hebb3,
                                                                              wins=time_window, label=labels, mask_dict = mask_dict[random_task_index])
                        targets_ = torch.zeros(batch_size, 10, device=device).scatter_(1, labels.view(-1, 1), 1)

                        surrogate_loss = 0.

                        loss = criterion(outputs, targets_ )
                        running_loss += loss.item()
                        loss.backward()
                        optim_local.step()
                        optim_local.zero_grad()
                        optim_base.zero_grad()

                        _, predicted = outputs.max(1)
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum()


            # Test the Model
            correct_all = []
            for task_index in range(train_task_index + 1):
                ss = total_random[task_index]
                correct = 0.
                total = 0.
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.float().to(device), labels.to(device)

                        numpy_data = images.view(batch_size, -1).data.cpu().numpy()
                        images = torch.from_numpy(numpy_data[:, ss]).cuda()

                        outputs, spikes, mems, _, _, _, _,_ = snn(input=images, hebb1=hebb1, hebb2=hebb2,hebb3=hebb3,
                                                                wins=time_window, label=labels, mask_dict = mask_dict[task_index])
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    correct_all.append((100 * float(correct) / float(total)))



            # total_avg.append(sum(correct_all) / len(correct_all))


            if train_task_index>=0:
                print("\n\n Completed %.2d,   Average Test Accuracy on All Tasks: {0:%.2f} "%
                      (train_task_index, sum(correct_all) / len(correct_all)))
            total_avg.append(sum(correct_all) / len(correct_all))

        state = {
            'net': snn.state_dict(),
            'total_avg': total_avg,
            'total_random': total_random
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + saving_name + '.t7')


def main():
    train()


if __name__ == '__main__':
    main()





