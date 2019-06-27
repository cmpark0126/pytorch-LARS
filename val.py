'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse

from hyperparams import Hyperparams_for_val as hp
from utils import progress_bar

with torch.cuda.device(hp.device[0]):
    # Data
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = models.resnet50()
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=hp.device)
    cudnn.benchmark = True

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(hp.checkpoint_folder_name), 'Error: no checkpoint directory found!'
    if hp.with_lars:
        checkpoint = torch.load('./' + hp.checkpoint_folder_name + '/withLars-' + str(hp.batch_size) + '.pth')
    else:
        checkpoint = torch.load('./' + hp.checkpoint_folder_name + '/noLars-' + str(hp.batch_size) + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    time_to_train = checkpoint['time_to_train'] # after 2nd 
    basic_info = checkpoint['basic_info'] # after 3rd

    criterion = nn.CrossEntropyLoss()

    def test():
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    if hp.with_lars:
        print('Resnet50, data=cifar10, With LARS, Validation')
    else:
        print('Resnet50, data=cifar10, Without LARS, Validation')
    print('basic_info=' + str(basic_info))

    for epo, acc, time in zip(epoch, best_acc, time_to_train):
        print (str(epo) + ' epoch | ' + str(acc) + ' % | ' + str(time) + ' sec')
    
    test()
    

