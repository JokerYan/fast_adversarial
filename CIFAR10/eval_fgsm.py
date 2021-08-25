import argparse
import logging
import os

import torch

from preact_resnet import PreActResNet18
from utils import evaluate_pgd, evaluate_standard, get_loaders, get_train_loaders_by_class, evaluate_pgd_post

pretrained_model_path = os.path.join('.', 'pretrained_models', 'cifar_model_weights_30_epochs.pth')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    state_dict = torch.load(pretrained_model_path)

    train_loader, test_loader = get_loaders(args.data_dir, batch_size=1)
    train_loaders_by_class = get_train_loaders_by_class(args.data_dir, batch_size=128)
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(state_dict)
    model_test.float()
    model_test.eval()

    # pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    pgd_loss, pgd_acc, pgd_loss_post, pgd_acc_post, test_loss, test_acc = evaluate_pgd_post(test_loader, train_loaders_by_class, model_test, 50, 10)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t PGD Post Loss \t PGD Post Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc, pgd_loss_post, pgd_acc_post)


if __name__ == '__main__':
    main()
