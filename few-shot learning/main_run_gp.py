'''
Main function of GP-based model
You should download the python version of the omniglot dataset from (https://github.com/brendenlake/omniglot) and
    modify the data_path
'''

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from gp_model import *
import time
import skimage
from skimage import io
import numpy as np
import glob
saving_name = '.'
#################################################### Modify the data path
data_path = '.' # Read the image datasest. # You should download the dataset, uncompress it and input the corresponding data_path.

print('compared with main_run_hp, to test the parameters')
def train():
    '''
    Training the HP-based models
    The parameter configuraions are listed in the *_model.py
    '''
    epoch = 0
    best_acc = 0
    print("Loading the Omniglot dataset...")

    def data_loader(data_path):
        # Read the image datasest .
        # You should download the dataset, uncompress it and input the corresponding data_path.
        imagedata = []
        for basedir in ('./dataset/omniglot-master/python/images_background/',
                       './dataset/omniglot-master/python/images_evaluation/'):
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

    imagedata = data_loader(data_path)
    np.random.shuffle(imagedata)  # Randomize order of characters
    print("Initializing network")
    net = hybridNet().to(device)
    w_list, meta_list = net.parameter_split()
    optimizer_meta = torch.optim.Adam(meta_list, lr=1.0 * meta_lr, eps = 1e-6)
    optimizer_w = torch.optim.Adam(w_list, lr=1.0 * gp_lr, eps = 1e-6)

    scheduler_meta = torch.optim.lr_scheduler.StepLR(optimizer_meta, step_size=steplr, gamma=gamma)
    scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, step_size=steplr, gamma=gamma)

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
            loss_meta = 0.
            x_spts, y_spts, t_spts, x_qrys, y_qrys, t_qrys = train_data_batch(imagedata)
            for i_task in range(num_train_task):
                # x_spt, y_spt, t_spt, x_qry, y_qry, t_qry = train_data_generation(imagedata)
                y, hebb = net(x_spts[i_task], y_spts[i_task], hebb)
                loss = criterion(y, t_spts[i_task])
                loss.backward()
                optimizer_w.step()
                optimizer_w.zero_grad()
                # yq, hebb = net(x_qry[i_task], y_qry, hebb)
                # lossq = criterion(yq, t_qry)
                # loss_meta += lossq

            buf = y.max(dim=1)[1]
            correct = float(buf.float() == (t_spts[i_task].max(dim=0)[1]).item())
            train_acc.append(correct)

            optimizer_meta.zero_grad()
            optimizer_w.zero_grad()

            # for i_task in range(num_val_task):
            #     # x_spt, y_spt, t_spt, x_qry, y_qry, t_qry = train_data_generation(imagedata)
            #     y, hebb = net(x_qrys[i_task], y_qrys[i_task], hebb)
            #     loss  = criterion(y, t_qrys[i_task])
            #     v_meta = tuple(net.meta)
            #     v_weights = tuple(net.weights)
            #     v_grads = torch.autograd.grad(loss, v_meta + v_weights, allow_unused=True)
            #     dmeta = v_grads[:(len(v_meta))]
            #     dw = v_grads[len(v_meta):]
            #     hessian = net.compute_hessian(dw, x_spts[i_task], y_spts[i_task], t_spts[i_task], criterion)
            #     with torch.no_grad():
            #         for alpha, da, h in zip(net.meta, dmeta, hessian):
            #             alpha.grad += da - lr_second * h
            #
            # nn.utils.clip_grad_norm_(meta_list, max_norm=1.)
            # optimizer_meta.step()
            # optimizer_meta.zero_grad()
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
                inputs, labels, target = test_data_generation(imagedata)
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

