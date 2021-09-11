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
    parser.set_defaults(mixup=True, type=bool)
    parser.add_argument('--no-mixup', dest='mixup', action='store_false')
    parser.add_argument('--pt-data', default='ori_rand', choices=['ori_rand', 'ori_train', 'rand'], type=str)
    parser.add_argument('--pt-method', default='adv', choices=['adv', 'normal'], type=str)
    parser.add_argument('--pt-iter', default=5, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    print(args)
    state_dict = torch.load(pretrained_model_path)

    _, test_loader = get_loaders(args.data_dir, batch_size=1)
    train_loader, _ = get_loaders(args.data_dir, batch_size=128)
    train_loaders_by_class = get_train_loaders_by_class(args.data_dir, batch_size=128)
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(state_dict)
    model_test.float()
    model_test.eval()

    # pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    pgd_loss, pgd_acc, pgd_loss_post, pgd_acc_post, normal_loss_post, normal_acc_post \
        = evaluate_pgd_post(test_loader, train_loader, train_loaders_by_class, model_test, 50, 10, args)

    logger.info('Normal Loss \t Normal Acc \t PGD Loss \t PGD Acc \t PGD Post Loss \t PGD Post Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f', normal_loss_post, normal_acc_post, pgd_loss, pgd_acc, pgd_loss_post, pgd_acc_post)


if __name__ == '__main__':
    main()
