import torch
import torch.nn as nn
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import random
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
import os

######################################## few-shot learning parameters
num_update = 2 # The update interval of meta-local parameters
num_task = 3 # Number of learning tasks, here we use a small number to accelerate training
num_test_classes = 400 # testing samples
steplr = 500000
nbclasses = 5  # Number of classifications
gamma = .5  # Decay factors of learning rate
num_shots = 1 # Number of shot in each episode
prestime = 1
mem_decay = 0.1 # Membrane potential factors
w_decay = 0.9
temperature = 0.1
hidden_num = 64  # Number of hidden feature
imgsize = 31 # Size of Resized image
nbiter = 5000000
meta_lr = 5e-5 #  learning rate of meta-local module
gp_lr = 3e-4 #  learning rate of gp module
ntest = 2500 # Reporting epoch
thr = 15 # Threshold
p = 0.0 # dropouts
input_dim = 512
wins = prestime * ( num_shots * nbclasses + 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < 0.5
        return grad_input * temp.float()

# Network parameters 
cfg_cnn = [(1, 128, 1, 3, 2),
           (128, 128, 1, 3, 2),
           (128, 128, 1, 3, 2),
           (128, 128, 1, 3, 2),]

fn1 = torch.tanh
fn2 = ActFun.apply

class hybridNet(nn.Module):
    def __init__(self, params=None):
        super(hybridNet, self).__init__()
        # Use a similar structure with previous work to abstract features
        in_planes, out_planes, padding, kernelsize,stride = cfg_cnn[0]
        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size =kernelsize, stride=stride, padding=padding)
        in_planes, out_planes, padding, kernelsize, stride = cfg_cnn[1]
        self.conv2 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding)
        in_planes, out_planes, padding, kernelsize, stride = cfg_cnn[2]
        self.conv3 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding)
        in_planes, out_planes, padding, kernelsize, stride = cfg_cnn[3]
        self.conv4 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding)

        # self.fc1 = torch.nn.Linear(hidden_num, hidden_num, bias=False)
        self.w1 = torch.nn.Parameter((1e-4 * torch.rand(input_dim, nbclasses)).cuda(), requires_grad=True)
        # Local parameters
        self.u1 = torch.nn.Parameter((1e-4 * torch.rand(input_dim, nbclasses)).cuda(), requires_grad=True)
        self.beta = torch.nn.Parameter((torch.ones(1)).cuda(), requires_grad=True)
        self.alpha = torch.nn.Parameter((1e-2 * torch.rand(input_dim, nbclasses)).cuda(), requires_grad=True)
        self.eta = torch.nn.Parameter((.05 * torch.ones(1)).cuda(), requires_grad=True)
        self.hidden_num = input_dim
        self.nbclasses = nbclasses

    def init_hebb(self):
        return torch.zeros(input_dim, nbclasses, device=device)

    def parameter_split(self):
        weight_param = []
        for n, p in self.named_parameters():
            if n[:2] == 'fc' or n[:2] == 'co':
                weight_param.append(p)
        local_param = list(set(self.parameters()) - set(weight_param))
        return weight_param, local_param

    def forward(self, inputs, labels, hebb):
        mem = spike = torch.zeros(1, self.nbclasses).cuda()
        hebb = torch.zeros(input_dim, self.nbclasses).cuda()
        for step in range(wins):
            inputx = inputs[step]
            # Here we follow a similar structure using in other reported works to abstract images feature 
            h1 = fn1(self.conv1(inputx))
            h1 = F.dropout(h1, p, training = self.training)
            h2 = fn1(self.conv2(h1))
            h2 = F.dropout(h2, p, training=self.training)
            h3 = fn1(self.conv3(h2))
            h3 = F.dropout(h3, p, training=self.training)
            h4 = fn1(self.conv4(h3))
            x = h4.view(-1, input_dim)
            # GP-based learning, here we do not add any local module .hmuwy
            # mem, spike, hebb = mem_update_plastic(self.w1, self.alpha, self.beta, self.eta, x, inputlabel, mem, spike,
            #                                       hebb)
            mem, spike = mem_update_no_plastic(self.w1, x,  mem, spike)

        outs = F.sigmoid(mem) # Use the sigmoid function to represent the firing rate of output neurons
        return outs, hebb


def mem_update_no_plastic(w,   x, mem, spike):
    stimulus = x.mm(w)  
    mem = (mem - thr * spike) * mem_decay + stimulus
    spike = fn2(mem - thr)
    return mem, spike 

# def mem_update_plastic(w, alpha, beta, eta, x, inputlabel, mem, spike, hebb):
#     stimulus = x.mm(torch.mul(alpha, w) + torch.mul(1 - alpha, hebb)) + beta * 10 * inputlabel
#     mem = (mem - thr * spike) * mem_decay + stimulus
#     spike = fn2(mem - thr)
#     hebb = hebb * w_decay + eta * torch.bmm(x.unsqueeze(2), F.softmax(mem / temperature, dim=1).unsqueeze(1)).mean( dim=0)
#     return mem, spike, hebb


# Generate one episoide of few-shot learning  
def data_generation(raw_data, test_mode=False):
    # test_mode = test
    transform_train = transforms.Compose([
        transforms.Resize((31, 31)),
        # transforms.RandomHorizontalFlip(0.2),
        # transforms.RandomRotation(degrees=(0, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.9195], std=[0.2721])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((31, 31)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.9195], std=[0.2721])
    ])

    images = torch.zeros((wins, 1, 1, imgsize, imgsize), device=device)  # (wins, 1, 1, imgsize, imgsize)
    labels = torch.zeros((wins, 1, nbclasses), device=device)  # (wins, 1, nbclasses)
    patterns = []
    ## generate the index array, named by class_index
    if test_mode:
        class_index = np.random.permutation(np.arange(len(raw_data) - num_test_classes, len(raw_data)))[:nbclasses]
    else:
        class_index = np.random.permutation(np.arange(len(raw_data) - num_test_classes))[:nbclasses]
    class_index = np.random.permutation(class_index)
    testcat = random.choice(class_index)  ## select the class on which we'll test in this episode
    copy_index = class_index.copy()

    # Inserting the character images and labels in the input tensor at the proper places
    location = 0
    for nc in range(num_shots):
        np.random.shuffle(class_index)  # Presentations occur in random order
        for ii, catnum in enumerate(class_index):

            img = random.choice(raw_data[catnum])
            if test_mode:
                img = transform_train(Image.fromarray(img))
            else:
                img = transform_test(Image.fromarray(img))

            images[location][0][0][:][:] = img[:][:]
            labels[location][0][np.where(copy_index == catnum)] = 1
            location += 1

    img = random.choice(raw_data[testcat])
    if test_mode:
        img = transform_train(Image.fromarray(img))
    else:
        img = transform_test(Image.fromarray(img))
    for nn in range(prestime):
        images[location][0][0][:][:] = img[:][:]
        location += 1
    # Generating the test label
    testlabel = np.zeros(nbclasses)
    testlabel[np.where(copy_index == testcat)] = 1
    assert (location == wins)
    targets = torch.from_numpy(testlabel).float().to(device)
    return images, labels, targets

