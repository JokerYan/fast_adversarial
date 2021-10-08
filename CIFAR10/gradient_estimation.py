import copy
import argparse

import torch
import torch.nn as nn
import numpy as np

from post_model import PostModel
from utils import get_loaders


repeat_count = 5
step_size = 0.0001
pixel_x = 16
pixel_y = 16
pixel_c = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.set_defaults(mixup=True, type=bool)
    parser.add_argument('--no-mixup', dest='mixup', action='store_false')
    parser.add_argument('--pt-data', default='ori_neigh', choices=['ori_rand', 'ori_train', 'ori_neigh_train', 'ori_neigh', 'rand'], type=str)
    parser.add_argument('--pt-method', default='adv', choices=['adv', 'normal'], type=str)
    parser.add_argument('--pt-iter', default=5, type=int)
    parser.set_defaults(rs_neigh=True, type=bool)
    parser.add_argument('--no-rs-neigh', dest='rs_neigh', action='store_false')
    parser.set_defaults(blackbox=False, type=bool)
    parser.add_argument('--blackbox', dest='blackbox', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    post_model = PostModel(model=None, args=args)
    _, test_loader = get_loaders(args.data_dir, batch_size=1)
    loss_func = nn.CrossEntropyLoss()
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        unit_error = torch.zeros_like(images)
        unit_error[0][pixel_c][pixel_x][pixel_y] = 1
        # for j in range(repeat_count):
        #     images_pos = copy.deepcopy(images).detach() + unit_error * step_size
        #     images_neg = copy.deepcopy(images).detach() - unit_error * step_size
        #     output_pos = post_model(images_pos).detach()
        #     output_neg = post_model(images_neg).detach()
        #     loss_pos = loss_func(output_pos, labels)
        #     loss_neg = loss_func(output_neg, labels)
        #     gradient = (loss_pos - loss_neg) / (2 * step_size)
        #     print("post gradient:", gradient)
        # for j in range(repeat_count):
        #     images_pos = copy.deepcopy(images).detach() + unit_error * step_size
        #     images_neg = copy.deepcopy(images).detach() - unit_error * step_size
        #     output_pos = post_model(images_pos, post=False).detach()
        #     output_neg = post_model(images_neg, post=False).detach()
        #     loss_pos = loss_func(output_pos, labels)
        #     loss_neg = loss_func(output_neg, labels)
        #     gradient = (loss_pos - loss_neg) / (2 * step_size)
        #     print("normal gradient:", gradient)

        # gradient gt
        images.requires_grad = True
        output = post_model(images, post=False)
        loss = loss_func(output, labels)
        all_gradient = torch.autograd.grad(loss, images)[0]
        print("gt gradient:", all_gradient[0][pixel_c][pixel_x][pixel_y])
        break


if __name__ == '__main__':
    main()