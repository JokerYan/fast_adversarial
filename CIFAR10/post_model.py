import argparse

import torch
import torch.nn as nn

from utils import get_loaders, get_train_loaders_by_class, post_train


class PostModel(nn.Module):
    def __init__(self, model, args=None):
        super().__init__()
        self.model = model

        if args is None:
            parser = argparse.ArgumentParser()
            args = parser.parse_args()
        self.args = args

        self.train_loader, _ = get_loaders(self.args.data_dir, batch_size=128)
        self.train_loaders_by_class = get_train_loaders_by_class(self.args.data_dir, batch_size=128)

    def forward(self, images):
        post_model, original_class, neighbour_class, loss_list, acc_list, neighbour_delta = \
            post_train(self.model, images, self.train_loader, self.train_loaders_by_class, self.args)
        return post_model(images)
