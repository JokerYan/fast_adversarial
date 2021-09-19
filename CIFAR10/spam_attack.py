import argparse
import logging
import os

import torch

from preact_resnet import PreActResNet18
from utils import evaluate_pgd, evaluate_standard, get_loaders, get_train_loaders_by_class, evaluate_pgd_post, \
    attack_pgd

pretrained_model_path = os.path.join('.', 'pretrained_models', 'cifar_model_weights_30_epochs.pth')
logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()
epsilon = (8 / 255.) / std
alpha = (2 / 255.) / std

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    print(args)
    state_dict = torch.load(pretrained_model_path)

    _, test_loader = get_loaders(args.data_dir, batch_size=1)
    model = PreActResNet18().cuda()
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    for images, label in test_loader:
        images = images.cuda()
        label = label.cuda()
        for i in range(1000):
            image_delta = attack_pgd(model, images, label, epsilon, alpha, 20, 1).detach()
            adv_output = model(images + image_delta)
            adv_class = torch.argmax(adv_output, dim=1).reshape(1)
            print(int(adv_class))
        print()


if __name__ == '__main__':
    main()
