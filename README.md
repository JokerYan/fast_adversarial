# Adaptive Modeling Against Adversarial Attacks (Fast FGSM Experiment)

This is the official experiment repo on Fast-FGSM base models for the [paper](https://arxiv.org/abs/2112.12431) "Adaptive Modeling Against Adversarial Attacks".

The main code repo: https://github.com/JokerYan/post_training
The Madry Model experiment repo: https://github.com/JokerYan/pytorch-adversarial-training.git 
The original Fast FGSM base model repo: https://github.com/locuslab/fast_adversarial

* Please note that the algorithm might be referred as **post training** for easy reference.

## Envrionment Setups
We recommend using Anaconda/Miniconda to setup the environment, with the following command:
```bash
conda env create -f pt_env.yml
conda activate post_train
```

## Experiments

### CIFAR-10
#### Base Model
The base model provided by the Fast-FGSM author can be found on the readme [here](https://github.com/locuslab/fast_adversarial/tree/master/CIFAR10).

#### Attack algorithm
20-step l infinity PGD without restart, with ϵ = 8/255 and step size α = 3/255

#### Post Training setups
* 50 epochs
* Batch Size 128
* SGD optimizer of 0.001, momentum of 0.9

#### Results
| Model | Robust Accuracy | Natural Accuracy |
| ----- | --------------- | ---------------- |
| Fast FGSM | 0.4681 | 0.8380 |
| Fast FGSM + Post Train (Fast) | 0.6127 | 0.8244 |
| Fast FGSM + Post Train (Fix Adv) | **0.6448** | 0.8556 |

#### How to Run
Please change the directory to `CIFAR10` before the experiment.
You can refer to the `bash` folder for various examples of bash files that runs the experiments. 
The experiment results will be updated in the respective log file in the `logs` folder.

Here is an example of experiment bash command:
```bash
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=0 python eval_fgsm.py \
  --pt-data ori_neigh \
  --pt-method adv \
  --adv-dir na \
  --neigh-method untargeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 20 \
  --att-restart 1 \
  --log-file logs/log_exp01_${TIMESTAMP}.txt
```

### MNIST
#### Base Model
The base model is already included by the original author of Fast-FGSM in `MNIST/models/fgsm.pth`.

#### Attack Algorithm
40-step l infinity PGD without restart, with ϵ = 0.3 and step size α = 0.01, unless otherwise specified

#### Post Training setups
* 50 epochs
* Batch Size 100
* SGD optimizer of 0.001, momentum of 0.9

#### Results
| Model | Robust Accuracy | Natural Accuracy |
| ----- | --------------- | ---------------- |
| Fast FGSM | 0.9204 | 0.9850 |
| Fast FGSM + Post Train (Fast) | 0.9413 | 0.9847 |
| Fast FGSM + Post Train (Fix Adv) | 0.9441 | 0.9846 |

#### How to Run
Please change the directory to `MNIST` before the experiment.
You can refer to the `bash` folder for various examples of bash files that runs the experiments. 
The experiment results will be updated in the respective log file in the `logs` folder.

Here is an example of experiment bash command:
```bash
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=0 python evaluate_mnist_post.py \
  --pt-data ori_neigh \
  --pt-method adv \
  --adv-dir na \
  --neigh-method untargeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 40 \
  --att-restart 1 \
  --log-file logs/log_exp01_${TIMESTAMP}.txt
```

### ImageNet
The code for ImageNet is not implemented yet.
  
### Arguments
Some bash arguments are shared between CIFAR-10 and MNIST experiments.

The arguement description and accepted values are listed here:
* pt-data: post training data composition
  - ori_rand: 50% original class + 50% random class
  - ori_neigh: 50% original calss + 50% neighbour class
  - train: random training data
* pt-method: post training method
  - adv: fast adversarial training used in Fast FGSM
  - dir_adv: fixed adversarial training proposed in paper
  - normal: normal training instead of adversarial training
* adv-dir: direction of fixed adversarial training
  - na: not applicable, used for adv and normal pt-method
  - pos: positive direction, data + fix perturbation
  - neg: negative direction, data - fix perturbation
  - both: default for dir_adv, random mixture of positive and negative direction
* neigh-method: attack method to find the neighbour
  - untargeted: use untargeted attack
  - targeted: use targeted attack and choose the highest confidence class
* pt-iter: post training iteration
* pt-lr: post training learning rate
* att-iter: attack iteration used for attack and post adversarial training
* att-restart: attack restart used for attack and post adversarial training
* log-file: log file stored path
