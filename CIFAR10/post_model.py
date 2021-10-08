import argparse
import os

import torch
import torch.nn as nn

from preact_resnet import PreActResNet18
from utils import get_loaders, get_train_loaders_by_class, post_train

pretrained_model_path = os.path.join('.', 'pretrained_models', 'cifar_model_weights_30_epochs.pth')

class DummyArgs:
    def __init__(self):
        self.data_dir = '../../cifar-data'
        self.mixup = False
        self.pt_data = 'ori_neigh'
        self.pt_method = 'adv'
        self.pt_iter = 50
        self.rs_neigh = False
        self.blackbox = False

class PostModel(nn.Module):
    def __init__(self, model=None, args=None, post=True):
        super().__init__()
        self.post = post

        if model is None:
            state_dict = torch.load(pretrained_model_path)
            model = PreActResNet18().cuda()
            model.load_state_dict(state_dict)
            model.float()
            model.eval()
        self.model = model

        if args is None:
            args = DummyArgs()
        self.args = args

        self.train_loader, _ = get_loaders(self.args.data_dir, batch_size=128)
        self.train_loaders_by_class = get_train_loaders_by_class(self.args.data_dir, batch_size=128)

    def forward(self, images, post=True):
        if self.post and post:
            post_model, original_class, neighbour_class, loss_list, acc_list, neighbour_delta = \
                post_train(self.model, images, self.train_loader, self.train_loaders_by_class, self.args)
        else:
            post_model = self.model
        return post_model(images)
