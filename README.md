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

### Hyperparams

-   Base

    -   batch_size: 기준 Batch size. 실험에서 사용되는 모든 Batch size는 이 size의 배수 형태로 나타난다.
    -   lr: 기준 learning rate. 일반적으로 linear scailing에서 기준 값으로 사용한다.
    -   multiples: 아래에서 설면되는 k를 구하기 위한 지수로 사용되는 배수이다.

-   Hyperparams

    -   batch_size: 실제 학습에서 사용하는 batch size
    -   lr: 실제 학습에서 초기 값으로 사용하는 learning rate

    -   momentum
    -   weight_decay
    -   trust_coef: trust coefficient로 LARS 사용시에 내부에서 구해지는 Local LR의 신뢰도를 의미

    -   warmup_multiplier
    -   warmup_epoch

    -   max_decay_epoch: polynomial decay를 최대한 진행할 epoch 수
    -   end_learning_rate: decay 작업이 모두 완료되었을 때 learning rate가 수렴될 값

    -   num_of_epoch: 학습을 돌릴 총 epoch 수

    -   with_lars

### Demonstration

-   Terminology
    -   k
        -   we increase the batch B by k
        -   start batch size is 128
        -   if we use 256 as batch size, k is 2 in this time
        -   `k = (2 ** (multiples - 1))`
    -   (nan) = nan 발생

#### Attempt 1 (first trial)

-   Configuration

    -   Hyperparams

        -   momentum = 0.9
        -   weigth_decay
            -   noLars -> 5e-04
            -   withLARS -> 5e-03
        -   warm-up for 5 epoch
            -   multiplier = k
            -   target lr follows linear scailing rule
        -   polynomial decay (power=2) LR policy (after warm-up)
            -   for 200 epoch
            -   minimum lr = 1.5e-05 \* k
        -   number of epoch = 200

-   Without LARS

| Batch | Base LR |    top-1 Accuracy, %   | Time to train |
| :---: | :-----: | :--------------------: | :-----------: |
|  128  |   0.15  | 89.15 %<br>(base line) |  2113.52 sec  |
|  256  |   0.15  |         88.43 %        |  1433.38 sec  |
|  512  |   0.15  |         88.72 %        |  1820.35 sec  |
|  1024 |   0.15  |         87.96 %        |  1303.54 sec  |
|  2048 |   0.15  |         87.05 %        |  1827.90 sec  |
|  4096 |   0.15  |         78.03 %        |  2083.24 sec  |
|  8192 |   0.15  |         14.59 %        |  1459.81 sec  |

-   With LARS (trust coefficient = 0.1)

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.15  |      89.16 %      |  3203.54 sec  |
|  256  |   0.15  |      89.19 %      |  2147.74 sec  |
|  512  |   0.15  |      89.29 %      |  1677.25 sec  |
|  1024 |   0.15  |      89.17 %      |  1604.91 sec  |
|  2048 |   0.15  |      88.70 %      |  1413.10 sec  |
|  4096 |   0.15  |      86.78 %      |  1609.08 sec  |
|  8192 |   0.15  |      80.85 %      |  1629.48 sec  |

#### Attempt 2

-   Configuration

    -   Hyperparams

        -   momentum = 0.9
        -   weigth_decay
            -   noLars -> 5e-04
            -   withLARS -> 5e-03
        -   warm-up for 5 epoch
            -   multiplier = 2 \* k
            -   target lr follows linear scailing rule
        -   polynomial decay (power=2) LR policy (after warm-up)
            -   for 200 epoch
            -   minimum lr = 1e-05
        -   number of epoch = 200

-   Without LARS

| Batch | Base LR |    top-1 Accuracy, %   | Time to train |
| :---: | :-----: | :--------------------: | :-----------: |
|  128  |   0.05  | 90.40 %<br>(base line) |  4232.56 sec  |
|  256  |   0.05  |         90.00 %        |  2968.43 sec  |
|  512  |   0.05  |         89.50 %        |  2707.79 sec  |
|  1024 |   0.05  |         89.27 %        |  2627.22 sec  |
|  2048 |   0.05  |         89.21 %        |  2500.02 sec  |
|  4096 |   0.05  |         84.73 %        |  2872.25 sec  |
|  8192 |   0.05  |         20.85 %        |  2923.95 sec  |

-   With LARS (trust coefficient = 0.1)

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.05  |      90.00 %      |  6792.61 sec  |
|  256  |   0.05  |      90.05 %      |  4506.06 sec  |
|  512  |   0.05  |      90.04 %      |  3329.19 sec  |
|  1024 |   0.05  |      90.11 %      |  2954.45 sec  |
|  2048 |   0.05  |      90.19 %      |  2773.21 sec  |
|  4096 |   0.05  |      88.49 %      |  2866.02 sec  |
|  8192 |   0.05  |   10.00 % (nan)   |     0 sec     |

#### Attempt 3

-   Configuration

    -   Hyperparams

        -   momentum = 0.9
        -   weigth_decay
            -   noLars -> 5e-04
            -   withLARS -> 5e-03
        -   warm-up for 5 epoch
            -   multiplier = 2
        -   polynomial decay (power=2) LR policy (after warm-up)
            -   for 200 epoch
            -   minimum lr = 1e-05 \* k
        -   number of epoch = 200

    -   Additional Jobs

        -   Use He initialization
        -   base lr은 미리 linear scailing rule에 따라 조정

-   Without LARS

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.05  |      89.76 %      |  3983.89 sec  |
|  256  |   0.1   |      90.08 %      |  3095.91 sec  |
|  512  |   0.2   |      89.34 %      |  2674.38 sec  |
|  1024 |   0.4   |      88.82 %      |  2581.19 sec  |
|  2048 |   0.8   |      89.29 %      |  2660.56 sec  |
|  4096 |   1.6   |      85.02 %      |  2871.04 sec  |
|  8192 |   3.2   |      77.72 %      |  3195.90 sec  |

-   With LARS (trust coefficient = 0.1)

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.05  |      90.11 %      |  6880.76 sec  |
|  256  |   0.1   |      90.12 %      |  4262.83 sec  |
|  512  |   0.2   |      90.11 %      |  3475.73 sec  |
|  1024 |   0.4   |      90.02 %      |  2760.31 sec  |
|  2048 |   0.8   |      90.02 %      |  2777.70 sec  |
|  4096 |   1.6   |      88.38 %      |  2946.53 sec  |
|  8192 |   3.2   |      86.40 %      |  3260.45 sec  |

#### Attempt 4

-   Configuration

    -   Hyperparams

        -   momentum = 0.9
        -   weigth_decay
            -   noLars -> 5e-04
            -   withLARS -> 5e-03
        -   warm-up for 5 epoch
            -   multiplier = 5
        -   polynomial decay (power=2) LR policy (after warm-up)
            -   for 200 epoch
            -   minimum lr = 1e-05 \* k
        -   number of epoch = 200

    -   Additional Jobs

        -   Use He initialization

-   Without LARS

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.02  |                   |               |
|  256  |   0.04  |                   |               |
|  512  |   0.08  |                   |               |
|  1024 |   0.16  |                   |               |
|  2048 |   0.32  |                   |               |
|  4096 |   0.64  |                   |               |
|  8192 |   1.28  |                   |               |

-   With LARS (trust coefficient = 0.1)

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.02  |                   |               |
|  256  |   0.04  |                   |               |
|  512  |   0.08  |                   |               |
|  1024 |   0.16  |                   |               |
|  2048 |   0.32  |                   |               |
|  4096 |   0.64  |                   |               |
|  8192 |   1.28  |                   |               |

#### Attempt 4

-   Configuration

    -   Hyperparams

        -   momentum = 0.9
        -   weigth_decay
            -   noLars -> 5e-04
            -   withLARS -> 5e-03
        -   warm-up for 5 epoch
            -   multiplier = 10
        -   polynomial decay (power=2) LR policy (after warm-up)
            -   for 200 epoch
            -   minimum lr = 1e-05 \* k
        -   number of epoch = 200

    -   Additional Jobs

        -   Use He initialization

-   Without LARS

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.01  |                   |               |
|  256  |   0.02  |                   |               |
|  512  |   0.04  |                   |               |
|  1024 |   0.08  |                   |               |
|  2048 |   0.16  |                   |               |
|  4096 |   0.32  |                   |               |
|  8192 |   0.64  |                   |               |

-   With LARS (trust coefficient = 0.1)

| Batch | Base LR | top-1 Accuracy, % | Time to train |
| :---: | :-----: | :---------------: | :-----------: |
|  128  |   0.01  |                   |               |
|  256  |   0.02  |                   |               |
|  512  |   0.04  |                   |               |
|  1024 |   0.08  |                   |               |
|  2048 |   0.16  |                   |               |
|  4096 |   0.32  |                   |               |
|  8192 |   0.64  |                   |               |

### Reference

-   Base code: <https://github.com/kuangliu/pytorch-cifar>
-   warm-up LR scheduler: <https://github.com/ildoonet/pytorch-gradual-warmup-lr/tree/master/warmup_scheduler>
    -   또한, 이를 기반으로 PolynomialLRDecay class 구현
        -   polynomial LR decay scheduler
    -   참고: scheduler.py
-   Pytorch Doc / Optimizer: <https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html>
    -   Optimizer class
    -   SGD class
