import argparse
import logging
import os

import torch

from preact_resnet import PreActResNet18
from utils import evaluate_pgd, evaluate_standard, get_loaders, get_train_loaders_by_class, evaluate_pgd_post, \
    get_blackbox_loader

pretrained_model_path = os.path.join('.', 'pretrained_models', 'cifar_model_weights_30_epochs.pth')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--pt-data', default='ori_neigh', choices=['ori_rand', 'ori_neigh', 'train'], type=str)
    parser.add_argument('--pt-method', default='adv', choices=['adv', 'dir_adv', 'normal'], type=str)
    parser.add_argument('--adv-dir', default='na', choices=['na', 'pos', 'neg', 'both'], type=str)
    parser.add_argument('--neigh-method', default='untargeted', choices=['untargeted', 'targeted'], type=str)
    parser.add_argument('--pt-iter', default=50, type=int)
    parser.add_argument('--pt-lr', default=0.001, type=float)
    parser.add_argument('--att-iter', default=20, type=int)
    parser.add_argument('--att-restart', default=1, type=int)
    parser.set_defaults(blackbox=False, type=bool)
    parser.add_argument('--blackbox', dest='blackbox', action='store_true')
    parser.add_argument('--log-file', default='logs/default.log', type=str)
    args = parser.parse_args()

    # check args validity
    if args.adv_dir != 'na':
        assert args.pt_method == 'dir_adv'
    if args.pt_method == 'dir_adv':
        assert args.adv_dir != 'na'
    return args


def main():
    args = get_args()
    # set logger file
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    logger.info(args)

    if not args.blackbox:
        _, test_loader = get_loaders(args.data_dir, batch_size=1)
    else:
        test_loader = get_blackbox_loader(batch_size=1)
    train_loader, _ = get_loaders(args.data_dir, batch_size=128)
    train_loaders_by_class = get_train_loaders_by_class(args.data_dir, batch_size=128)
    model_test = PreActResNet18().cuda()
    state_dict = torch.load(pretrained_model_path)
    model_test.load_state_dict(state_dict)
    model_test.float()
    model_test.eval()

    # pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    pgd_loss, pgd_acc, pgd_loss_post, pgd_acc_post, normal_loss_post, normal_acc_post \
        = evaluate_pgd_post(test_loader, train_loader, train_loaders_by_class, model_test, args)

    logger.info('Normal Loss \t Normal Acc \t PGD Loss \t PGD Acc \t PGD Post Loss \t PGD Post Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f', normal_loss_post, normal_acc_post, pgd_loss, pgd_acc, pgd_loss_post, pgd_acc_post)


if __name__ == '__main__':
    main()
