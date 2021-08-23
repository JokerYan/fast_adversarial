import argparse
import logging
import os

import torch

from CIFAR10.preact_resnet import PreActResNet18
from CIFAR10.utils import evaluate_pgd, evaluate_standard, get_loaders

pretrained_model_path = os.path.join('.', 'pretrained_models', 'cifar_model_weights_30_epochs.pth')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    state_dict = torch.load(pretrained_model_path)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(state_dict)
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == '__main__':
    main()
