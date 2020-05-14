'''
Main function of GP-based model
You should download the python version of the omniglot dataset from (https://github.com/brendenlake/omniglot) and
    modify the data_path
'''
from gp_model import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import time
import skimage
from skimage import io
import numpy as np
import glob
saving_name = '.'
# print('nbclass', nbclasses,'hidden',hidden_num, 'numtask:',num_task, 'prob.', p)
#################################################### Modify the data path
data_path = './python'


def data_loader(data_path):
    # Read the image datasest .
    # You should download the dataset, uncompress it and input the corresponding data_path.
    imagedata = []
    for basedir in (data_path + '/images_background/',
                    data_path + '/images_evaluation/'):
        filedirs = glob.glob(basedir + '*')
        for dir in filedirs:
            chardirs = glob.glob(dir + "/*")
            for chardir in chardirs:
                chardata = []
                charfiles = glob.glob(chardir + '/*')
                for fn in charfiles:
                    filedata = skimage.io.imread(fn) / 255.0
                    chardata.append(filedata)
                imagedata.append(chardata)
    return imagedata

def train():
    '''
    Training the HP-based models
    The parameter configuraions are listed in the *_model.py
    '''
    epoch = 0
    best_acc = 0
    print("Loading the Omniglot dataset...")
    imagedata = data_loader(data_path)
    np.random.shuffle(imagedata)  # Randomize order of characters
    print("Initializing network")
    net = hybridNet().to(device)
    w_list, meta_list = net.parameter_split()
    optimizer_meta = torch.optim.Adam(meta_list, lr=1.0 * meta_lr)
    optimizer_w = torch.optim.Adam(w_list, lr=1.0 * gp_lr)
    scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, gamma=gamma, step_size=steplr)
    scheduler_meta = torch.optim.lr_scheduler.StepLR(optimizer_meta, gamma=gamma, step_size=steplr)

    loss_record = []
    acc_record = []
    all_losses = []
    trainloss = 0.
    train_acc = []

    nowtime = time.time()
    criterion = torch.nn.BCELoss()
    print("Starting training...")
    for numiter in range(nbiter):
        hebb = net.init_hebb()
        is_test_mode = ((numiter + 1) % ntest == 0)
        if is_test_mode == False:
            net.train()
            net.zero_grad()
            optimizer_w.zero_grad()
            for i_task in range(2):
                inputs, labels, target = data_generation(imagedata, test_mode=is_test_mode)
                y, hebb = net(inputs, labels, hebb)
                loss = criterion(y, target)
                loss.backward()
                optimizer_w.step()
                optimizer_w.zero_grad()

            buf = y.max(dim=1)[1]
            correct = float(buf.float() == (target.max(dim=0)[1]).item())
            train_acc.append(correct)
            optimizer_meta.step()
            optimizer_meta.zero_grad()

        # scheduler update
        scheduler_w.step()
        scheduler_meta.step()

        trainloss += loss.item()
        all_losses.append(trainloss)
        if is_test_mode:
            net.eval()

            print('\n\nEPOCH : ', epoch)
            print('Training acc: %.4f Training loss :%.4f' % (np.mean(train_acc), trainloss/(ntest+1)))

            class_acc = []
            l1_loss = []
            train_acc = []
            trainloss = 0.
            epoch += 1

            for idx_test in range(num_test_classes):
                # Testing
                inputs, labels, target = data_generation(imagedata, test_mode=is_test_mode)
                y, hebb = net(inputs, labels, hebb)
                td = target.cpu().numpy()
                yd = y.data.cpu().numpy()[0]
                buf = y.max(dim=1)[1]
                correct = float(buf.float() == (target.max(dim=0)[1]).item())
                class_acc.append(correct)
                absdiff = np.abs(td - yd)
                l1_loss.append(np.mean(absdiff))
            print("L1 loss:", np.mean(l1_loss))
            mean_acc = np.mean(class_acc)

            acc_record.append(mean_acc)
            loss_record.append(l1_loss)

            state = {
                'net': net.state_dict(),
                'acc_record': acc_record,
                'loss_record': loss_record,
                'hebb': hebb,
            }

            print("Testing accuracy: ", mean_acc)

            if mean_acc > best_acc:
                best_acc = mean_acc
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt' + saving_name + '.t7')
                print('best results:', best_acc)

            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", ntest, "iters: ", nowtime - previoustime)


def main():
    train()


if __name__ == "__main__":
    train()
    # main()

