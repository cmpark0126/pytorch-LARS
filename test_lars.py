import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.models as models

import os
import time

from optimizer import SGD_without_lars, SGD_with_lars, SGD_with_lars_ver2
from scheduler import GradualWarmupScheduler, PolynomialLRDecay
from hyperparams import Hyperparams as hp
from utils import progress_bar

with torch.cuda.device(0):
   # Model
    print('==> Building model..')
    net = models.resnet50()
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True

   # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = SGD_with_lars(net.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay, trust_coef=hp.trust_coef)
    optimizer = SGD_with_lars_ver2(net.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay, trust_coef=hp.trust_coef)
    # optimizer = SGD_without_lars(net.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay)
    # optimizer = optim.SGD(net.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay)

    # Training
    net.train()

    inputs = torch.ones([2, 3, 32, 32]).cuda()
    targets = torch.ones([2], dtype=torch.long).cuda()
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    print('Complete Forward & Backward')

    for batch_idx in range(5):
        start_time = time.time()
        # torch.cuda.nvtx.range_push('trial')
        
        optimizer.step()

        # torch.cuda.nvtx.range_pop()
        print('time to optimize is %.3f' % (time.time() - start_time))
        
