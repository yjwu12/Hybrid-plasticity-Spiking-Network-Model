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
lr_second = 0.1 # learning rate of secord order deriviates
num_update = 1  # The update interval for alternatively  updating  meta-parameters and weights
num_train_task = 5  # Number of training tasks, here we use a small number to accelerate training
num_val_task = 1 # Number of validation tasks
num_test_classes = 100  # testing samples
nbclasses = 5  # Number of classifications
gamma = .5  # Decay factors of learning rate
num_shots = 1  # Number of shot in each episode
prestime = 1
mem_decay = 0.2  # Membrane potential factors
w_decay = 0.99
temperature = 0.1 # softmax coe-eficience
imgsize = 31  # Size of Resized image
ntest = 1000  # Reporting epoch
steplr = 10000
nbiter = 1000000

meta_lr = 3e-5  # learning rate of meta-local module
gp_lr = 8e-5  # learning rate of gp module
thr = 0.5  # Threshold
p = 0.0  # dropouts
lens = 0.5 # hyperparamters for firing function
w_binding = 10 # binding weights used in the last layers
input_dim = 512
wins = prestime * (num_shots * nbclasses + 1)
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
        temp = abs(input) < lens
        return grad_input * temp.float()


# Network parameters
cfg_cnn = [(1, 128, 1, 3, 2),
           (128, 128, 1, 3, 2),
           (128, 128, 1, 3, 2),
           (128, 128, 1, 3, 2), ]

fn1 = torch.tanh
fn2 = ActFun.apply

class hybridNet(nn.Module):
    def __init__(self, params=None):
        super(hybridNet, self).__init__()
        # Use a similar structure with previous work to abstract features
        in_planes, out_planes, padding, kernelsize, stride = cfg_cnn[0]
        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding)
        in_planes, out_planes, padding, kernelsize, stride = cfg_cnn[1]
        self.conv2 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding)
        in_planes, out_planes, padding, kernelsize, stride = cfg_cnn[2]
        self.conv3 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding)
        in_planes, out_planes, padding, kernelsize, stride = cfg_cnn[3]
        self.conv4 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding)
        # self.fc0 = torch.nn.Linear(input_dim, input_dim)
        self.fc1 = torch.nn.Linear(input_dim, nbclasses, bias=False)
        # Local parameters
        self.alpha = torch.nn.Parameter((1e-1 * torch.rand(input_dim, nbclasses)).cuda(), requires_grad=True)
        self.beta = torch.nn.Parameter((1e-1 * torch.rand(nbclasses)).cuda(), requires_grad=True)
        self.eta = torch.nn.Parameter((1e-1 * torch.ones(1)).cuda(), requires_grad=True)
        self.nbclasses = nbclasses
        self.weights = []
        self.meta = []

    def compute_hessian(self, dw, train_x, train_labels, train_tagets, criterion):
        """
        :param dw: dw = dw' {L_val(w', alpha)}
        :param train_y:
        :param train_label:
        :return:
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = min(1e-3 / norm, 1e-3)
        with torch.no_grad():
            for p, d in zip(self.weights, dw):
                p += eps * d

        train_y, hebb = self.forward(train_x, train_labels)
        loss = criterion(train_y, train_tagets)
        dmeta_pos = torch.autograd.grad(loss, self.meta, allow_unused=True)
        with torch.no_grad():
            for p, d in zip(self.weights, dw):
                p -= 2. * eps * d

        train_y, hebb = self.forward(train_x, train_labels)
        loss = criterion(train_y, train_tagets)
        dmeta_neg = torch.autograd.grad(loss, self.meta, allow_unused=True)

        with torch.no_grad():
            for p, d in zip(self.weights, dw):
                p += eps * d
        # hessian = [(p - n) / (2. * eps) for p, n in zip(dmeta_pos, dmeta_neg)]
        hessian = [torch.clamp((p - n) / (2. * eps), min=-1, max=1) for p, n in zip(dmeta_pos, dmeta_neg)]
        return hessian

    def init_hebb(self):
        return torch.zeros(input_dim, nbclasses, device=device)

    def parameter_split(self):
        '''
        Split the parameters of GP-based and LP-based learning.
        These two types of parameters have different update manner.
        '''
        weight_param = []
        for n, p in self.named_parameters():
            if n[:2] == 'fc' or n[:2] == 'co':
                weight_param.append(p)
        local_param = list(set(self.parameters()) - set(weight_param))
        self.weights = weight_param
        self.meta = local_param
        return weight_param, local_param

    def forward(self, inputs, labels, hebb=None):
        mem0 = spike0 = torch.zeros(1, input_dim).cuda()
        mem1 = spike1 = torch.zeros(1, self.nbclasses).cuda()
        hebb = torch.zeros(input_dim, self.nbclasses).cuda()
        for step in range(wins):
            inputx = inputs[step]
            inputlabel = labels[step]
            # Here we follow a common used structure using in other reported works to abstract features
            h1 = fn1(self.conv1(inputx))
            h1 = F.dropout(h1, p, training=self.training)
            h2 = fn1(self.conv2(h1))
            h2 = F.dropout(h2, p, training=self.training)
            h3 = fn1(self.conv3(h2))
            h3 = F.dropout(h3, p, training=self.training)
            h4 = fn1(self.conv4(h3))
            # Note : we can use an encoding layer to convert features into spike trains, but we ignore it here for simplicity.
            # mem0, spike0 = mem_update_no_plastic(self.fc0, x, mem0, spike0)
            spike0 = h4.view(-1, input_dim)

            mem1, spike1, hebb = mem_update_plastic(self.fc1, self.alpha, self.beta, self.eta, spike0, inputlabel, mem1,
                                                    spike1, hebb)
        outs = F.softmax(mem1, dim=1)  # Use the sigmoid function to represent the firing rate of output neurons
        return outs, hebb


def mem_update_no_plastic(fc, x, mem, spike):
    stimulus = fc(x)
    mem = (1 - spike) * mem * mem_decay + stimulus
    spike = fn2(mem - thr)
    mem.clamp_(min=- thr, max=thr)
    return mem, spike


def mem_update_plastic(w, alpha, beta, eta, x, inputlabel, mem, spike, hebb):
    stimulus = x.mm(torch.mul(alpha, w.weight.t()) + torch.mul(1 - alpha, hebb)) + w_binding * inputlabel
    mem = (1 - spike) * mem * mem_decay + stimulus
    spike = fn2(mem - thr)
    # hebb = hebb * w_decay + eta.clamp(max=0.1) * torch.bmm(x.unsqueeze(2),
    #                                                        F.softmax(mem - beta, dim=1).unsqueeze(1)).mean(dim=0)
    return mem, spike, hebb


def test_data_generation(task_set):
    transform_test = transforms.Compose([
        transforms.Resize((31, 31)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.9195], std=[0.2721])
    ])

    images = torch.zeros((wins, 1, 1, imgsize, imgsize), device=device)  # (wins, 1, 1, imgsize, imgsize)
    labels = torch.zeros((wins, 1, nbclasses), device=device)  # (wins, 1, nbclasses)
    patterns = []
    ## generate the index array, named by class_index
    class_index = np.random.permutation(np.arange(len(task_set) - num_test_classes, len(task_set)))[:nbclasses]

    class_index = np.random.permutation(class_index)
    testcat = random.choice(class_index)  ## select the class on which we'll test in this episode
    copy_index = class_index.copy()

    # Inserting the character images and labels in the input tensor at the proper places
    index = 0
    for nc in range(num_shots):
        np.random.shuffle(class_index)  # Presentations occur in random order
        for ii, catnum in enumerate(class_index):
            img = random.choice(task_set[catnum])
            img = transform_test(Image.fromarray(img))
            images[index][0][0][:][:] = img[:][:]
            labels[index][0][np.where(copy_index == catnum)] = 1
            index += 1

    img = random.choice(task_set[testcat])
    img = transform_test(Image.fromarray(img))
    for nn in range(prestime):
        images[index][0][0][:][:] = img[:][:]
        index += 1

    # Generating the test label
    testlabel = np.zeros(nbclasses)
    testlabel[np.where(copy_index == testcat)] = 1
    targets = torch.from_numpy(testlabel).float().to(device)
    return images, labels, targets


# Generate one episoide of few-shot learning
def train_data_generation(task_set):
    # test_mode = test
    transform_train = transforms.Compose([
        transforms.Resize((31, 31)),
        transforms.RandomHorizontalFlip(0.2),
        # transforms.RandomRotation(degrees=(0, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.9195], std=[0.2721])
    ])

    x_spt = torch.zeros((wins, 1, 1, imgsize, imgsize), device=device)  # (wins, 1, 1, imgsize, imgsize)
    x_qry = torch.zeros((wins, 1, 1, imgsize, imgsize), device=device)
    y_spt = torch.zeros((wins, 1, nbclasses), device=device)  # (wins, 1, nbclasses)
    y_qry = torch.zeros((wins, 1, nbclasses), device=device)  # (wins, 1, nbclasses)
    patterns = []
    ## generate the index array, named by class_index
    class_index = np.random.permutation(np.arange(len(task_set) - num_test_classes))[:nbclasses]
    class_index = np.random.permutation(class_index)
    # testcat = random.choice(class_index)  ## select the class on which we'll test in this episode
    selected_spt = selected_qry = random.choice(class_index)

    copy_index = class_index.copy()

    # Inserting the character images and labels in the input tensor at the proper places
    index = 0
    for nc in range(num_shots):
        np.random.shuffle(class_index)  # Presentations occur in random order
        for ii, catnum in enumerate(class_index):
            # select samples from the task_set
            img_spt = random.choice(task_set[catnum])
            img_qry = random.choice(task_set[catnum])
            img_spt, img_qry = transform_train(Image.fromarray(img_spt)), transform_train(Image.fromarray(img_qry))

            x_spt[index][0][0] = img_spt
            x_qry[index][0][0] = img_qry
            y_spt[index][0][np.where(copy_index == catnum)] = 1
            y_qry[index][0][np.where(copy_index == catnum)] = 1
            index += 1

    # img = random.choice(task_set[testcat])
    img_spt = transform_train(Image.fromarray(random.choice(task_set[selected_spt])))
    img_qry = transform_train(Image.fromarray(random.choice(task_set[selected_qry])))

    for nn in range(prestime):
        x_spt[index][0][0] = img_spt
        x_qry[index][0][0] = img_qry
        index += 1
    # Generating the test label
    target_spt = target_qry = np.zeros(nbclasses)

    target_spt[np.where(copy_index == selected_spt)] = 1
    target_spt = torch.from_numpy(target_spt).float().to(device)
    target_qry[np.where(copy_index == selected_qry)] = 1
    target_qry = torch.from_numpy(target_qry).float().to(device)

    return x_spt, y_spt, target_spt, x_qry, y_qry, target_qry


def train_data_batch(dataset, num_epoch=max(num_val_task, num_train_task)):
    x_spts = []
    y_spts = []
    t_spts = []
    x_qrys = []
    y_qrys = []
    t_qrys = []
    import copy

    for idx in range(num_epoch):
        x_spt, y_spt, t_spt, x_qry, y_qry, t_qry = train_data_generation(dataset)
        x_spts.append(copy.deepcopy(x_spt))
        y_spts.append(copy.deepcopy(y_spt))
        t_spts.append(copy.deepcopy(t_spt))
        x_qrys.append(copy.deepcopy(x_qry))
        y_qrys.append(copy.deepcopy(y_qry))
        t_qrys.append(copy.deepcopy(t_qry))

    return x_spts, y_spts, t_spts, x_qrys, y_qrys, t_qrys

