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
import time

from Optimizer import SGD_without_lars, SGD_with_lars, SGD_with_lars_ver2
from Scheduler import GradualWarmupScheduler, PolynomialLRDecay
from hyperparams import Hyperparams as hp
from utils import progress_bar

with torch.cuda.device(hp.device[0]):
    all_accs = []
    best_acc = 0  # best test accuracy
    all_epochs = []
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    all_times = []
    time_to_train = 0

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    num_of_mini_batch = 1 if hp.batch_size <= 8192 else hp.batch_size // 8192 # hp.batch_size must be multiply of 8192

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=hp.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = models.resnet50()
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=hp.device)
    cudnn.benchmark = True

    if hp.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        if hp.with_lars:
            checkpoint = torch.load('./checkpoint/withLars-' + str(hp.batch_size) + '.pth')
        else:
            checkpoint = torch.load('./checkpoint/noLars-' + str(hp.batch_size) + '.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        time_to_train = checkpoint['time_to_train']
        basic_info = checkpoint['basic_info']

    # Loss & Optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = None
    if hp.with_lars:
    #     optimizer = SGD_with_lars(net.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay, trust_coef=hp.trust_coef)
        optimizer = SGD_with_lars_ver2(net.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay, trust_coef=hp.trust_coef)
    else:
    #     optimizer = SGD_without_lars(net.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay)
        optimizer = optim.SGD(net.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay)

    warmup_scheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=hp.warmup_multiplier, total_epoch=hp.warmup_epoch)
    poly_decay_scheduler = PolynomialLRDecay(optimizer=optimizer, max_decay_steps=hp.max_decay_epoch * len(trainloader), 
                                             end_learning_rate=hp.end_learning_rate, power=2.0) # poly(2)
    # Training
    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        global time_to_train

        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if epoch > hp.warmup_epoch: # after warmup schduler step
                poly_decay_scheduler.step()
    #             for param_group in optimizer.param_groups:
    #                 print(param_group['lr'])
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        time_to_train = time_to_train + (time.time() - start_time)

    def test(epoch):
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

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            all_accs.append(acc)
            all_epochs.append(epoch)
            all_times.append(round(time_to_train, 2))
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': all_accs,
                'epoch': all_epochs,
                'time_to_train': all_times,
                'basic_info': hp.get_info_dict()
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if hp.with_lars:
                torch.save(state, './checkpoint/withLars-' + str(hp.batch_size) + '.pth')
            else:
                torch.save(state, './checkpoint/noLars-' + str(hp.batch_size) + '.pth')
            best_acc = acc

    if hp.with_lars:
        print('Resnet50, data=cifar10, With LARS')
    else:
        print('Resnet50, data=cifar10, Without LARS')
    hp.print_hyperparms()
    for epoch in range(0, hp.num_of_epoch):
        print('\nEpoch: %d' % epoch)
        if epoch <= hp.warmup_epoch: # for readability
            warmup_scheduler.step()
        if epoch > hp.warmup_epoch: # after warmup, start decay scheduler with warmup-ed learning rate
            poly_decay_scheduler.base_lrs = warmup_scheduler.get_lr()
    #     for param_group in optimizer.param_groups:
    #         print(param_group['lr'])
        train(epoch)
        test(epoch)

    
    
  



