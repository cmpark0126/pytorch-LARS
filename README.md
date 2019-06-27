# Implement LARS

### Objective

-   link: ["Large Batch Training of Convolutional Networks (LARS)"](https://arxiv.org/abs/1708.03888)
-   위 논문에 소개된 LARS를 PyTorch로 구현
-   Data: CIFAR10

### Usage

```bash
$ git clone https://github.com/cmpark0126/pytorch-LARS.git
$ cd pytorch-LARS/
$ vi hyperparams.py # hyperparameter 및 config용 요소 확인. 필요시 수정 가능
$ python train.py # CIFAR10 학습 시작
$ python val.py # 학습 결과 확인
```

### Demonstration

- Terminology
    -   k
        -   we increase the batch B by k
        -   start batch size is 128
        -   if we use 256 as batch size, k is 2 in this time
    -   (nan) = nan 발생

#### Attempt 1

-   Configuration

    -   Hyperparams

        -   momentum = 0.9
        -   weigth_decay = 5e-04 (LARS = 5e-03)
        -   warm-up for 5 epoch
            -   multiplier = 2 \* (2 ^ (k - 1))
            -   target lr follows linear scailing rule
        -   polynomial decay (power=2) LR policy (after warm-up)
            -   for 200 epoch
            -   minimum lr = 1e-05
        -   number of epoch = 200

-   Without LARS

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.05  |    (base line)    |               |
|  256  |   0.05  |                   |               |
|  512  |   0.05  |                   |               |
|   1K  |   0.05  |                   |               |
|   2K  |   0.05  |                   |               |
|   4K  |   0.05  |                   |               |
|   8K  |   0.05  |                   |               |

-   With LARS

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.05  |                   |               |
|  256  |   0.05  |                   |               |
|  512  |   0.05  |                   |               |
|   1K  |   0.05  |                   |               |
|   2K  |   0.05  |                   |               |
|   4K  |   0.05  |                   |               |
|   8K  |   0.05  |     10.0 (nan)    |       0       |

#### Attempt 2

-   Configuration

    -   Hyperparams

        -   momentum = 0.9
        -   weigth_decay = 5e-04 (LARS = 5e-03)
        -   warm-up for 5 epoch
            -   multiplier = 2
        -   polynomial decay (power=2) LR policy (after warm-up)
            -   for 200 epoch
            -   minimum lr = 1e-05 \* (2 ^ (k - 1))
        -   number of epoch = 200

    -   Additional Jobs

        -   Use He initialization

-   Without LARS

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.05  |                   |               |
|  256  |   0.1   |                   |               |
|  512  |   0.2   |                   |               |
|   1K  |   0.4   |                   |               |
|   2K  |   0.8   |                   |               |
|   4K  |   1.6   |                   |               |
|   8K  |   3.2   |                   |               |

-   With LARS

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.05  |                   |               |
|  256  |   0.1   |                   |               |
|  512  |   0.2   |                   |               |
|   1K  |   0.4   |                   |               |
|   2K  |   0.8   |                   |               |
|   4K  |   1.6   |                   |               |
|   8K  |   3.2   |                   |               |

#### Attempt 3

-   Configuration

    -   Hyperparams

        -   step LR lr_scheduler
            -   step size = 50
            -   gamma = 0.1

#### Attempt 4

-   Configuration

    -   Hyperparams

        -   step LR lr_scheduler
            -   step size = 50
            -   gamma = 0.1

    -   Additional Jobs

        -   Use He initialization

#### Attempt 5

### Reference

-   Base code: <https://github.com/kuangliu/pytorch-cifar>
-   warm-up LR scheduler: <https://github.com/ildoonet/pytorch-gradual-warmup-lr/tree/master/warmup_scheduler>
    -   또한, 이를 기반으로 PolynomialLRDecay class 구현
        -   polynomial LR decay scheduler
    -   참고: scheduler.py
-   Pytorch Doc / Optimizer: <https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html>
    -   Optimizer class
    -   SGD class
