import os
import logging
import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from preact_resnet import PreActResNet18
from utils import evaluate_pgd, evaluate_standard, get_loaders, get_train_loaders_by_class, evaluate_pgd_post, \
    get_blackbox_loader, attack_pgd

pretrained_model_path = os.path.join('.', 'pretrained_models', 'cifar_model_weights_30_epochs.pth')

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

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
    logger = logging.getLogger("eval")
    logger.info(args)

    if not args.blackbox:
        _, test_loader = get_loaders(args.data_dir, batch_size=1)
    else:
        test_loader = get_blackbox_loader(batch_size=1)
    train_loader, _ = get_loaders(args.data_dir, batch_size=128)
    model = PreActResNet18().cuda()
    state_dict = torch.load(pretrained_model_path)
    model.load_state_dict(state_dict)
    model.float()
    model.eval()

    # attack
    alpha = (10 / 255) / std
    epsilon = (8 / 255) / std
    n = 0
    pgd_loss = 0
    pgd_acc = 0

    adv_data_list = []
    label_list = []

    for i, (X, y) in enumerate(test_loader):
        n += y.size(0)
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, args.att_iter, args.att_restart).detach()
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            logger.info('Batch {}\tbase adv acc: {:.4f}'.format(i+1, pgd_acc / n))

            adv_data = X + pgd_delta
            adv_data = adv_data * std + mu  # denormalized
            adv_data_list.append(adv_data.detach().cpu())
            label_list.append(y.detach().cpu())

    adv_data_concat = torch.cat(adv_data_list, 0)
    label_concat = torch.cat(label_list, 0)

    saved_adv = (adv_data_concat, label_concat)
    saved_path = "../../data/cifar10_adv_fast.pickle"
    with open(saved_path, "wb") as f:
        pickle.dump(saved_adv, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("adv saved to path: {}".format(saved_path))

if __name__ == '__main__':
    main()
