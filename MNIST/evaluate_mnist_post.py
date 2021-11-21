import argparse
import copy
import logging
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset

from mnist_net import mnist_net

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def cal_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, 1)
    # collect the correct predictions for each class
    correct = 0
    total = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
        total += 1
    return correct / total


def merge_images(train_images, val_images, ratio, device):
    batch_size = len(train_images)
    repeated_val_images = val_images.repeat(batch_size, 1, 1, 1)
    merged_images = ratio * train_images.to(device) + (1 - ratio) * repeated_val_images.to(device)
    # image[0][channel] = 0.5 * image[0][channel].to(device) + 0.5 * val_images[0][channel].to(device)
    return merged_images


def attack_fgsm(model, X, y, epsilon):
    delta = torch.zeros_like(X, requires_grad=True)
    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = epsilon * torch.sign(grad)
    return delta.detach()


def get_train_loaders_by_class(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(
        dir_, train=True, transform=train_transform, download=True)
    indices_list = [[] for _ in range(10)]
    for i in range(len(train_dataset)):
        label = int(train_dataset[i][1])
        indices_list[label].append(i)
    dataset_list = [Subset(train_dataset, indices) for indices in indices_list]
    train_loader_list = [
        torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        ) for dataset in dataset_list
    ]
    return train_loader_list


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, random_start=True):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        if random_start:
            delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        else:
            delta = torch.zeros_like(X).cuda()
        delta.data = clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)[0]
            if len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            d = clamp(d, 0-X, 1-X)
            delta.data[index] = d[index]
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def attack_pgd_targeted(model, X, y, target, epsilon, alpha, attack_iters, restarts, random_start=True):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if random_start:
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, 0 - X, 1 - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] != target)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, target)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = -1 * grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), target, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def post_train(model, images, train_loader, train_loaders_by_class, args):
    logger = logging.getLogger("eval")

    alpha = (10 / 255)
    epsilon = (8 / 255)
    loss_func = nn.CrossEntropyLoss()
    device = torch.device('cuda')
    model = copy.deepcopy(model)
    # model.train()
    fix_model = copy.deepcopy(model)
    # attack_model = torchattacks.PGD(model, eps=(8/255)/std, alpha=(2/255)/std, steps=20)
    optimizer = torch.optim.SGD(lr=args.pt_lr,
                                params=model.parameters(),
                                momentum=0.9,
                                nesterov=True)
    images = images.detach()
    with torch.enable_grad():
        # find neighbour
        original_output = fix_model(images)
        original_class = torch.argmax(original_output).reshape(1)

        if args.neigh_method == 'targeted':
            # targeted attack to find neighbour
            min_target_loss = float('inf')
            max_target_loss = float('-inf')
            neighbour_delta = None
            for target_idx in range(10):
                if target_idx == original_class:
                    continue
                target = torch.ones_like(original_class) * target_idx
                neighbour_delta_targeted = attack_pgd_targeted(model, images, original_class, target, epsilon, alpha,
                                                               attack_iters=20, restarts=1, random_start=False).detach()
                target_output = fix_model(images + neighbour_delta_targeted)
                target_loss = loss_func(target_output, target)
                if target_loss < min_target_loss:
                    min_target_loss = target_loss
                    neighbour_delta = neighbour_delta_targeted
                # print(int(target), float(target_loss))
        elif args.neigh_method == 'untargeted':
            # neighbour_images = attack_model(images, original_class)
            neighbour_delta = attack_pgd(model, images, original_class, epsilon, alpha, attack_iters=20, restarts=1,
                                         random_start=False).detach()

        neighbour_images = neighbour_delta + images
        neighbour_output = fix_model(neighbour_images)
        neighbour_class = torch.argmax(neighbour_output).reshape(1)

        if original_class == neighbour_class:
            logger.info('original class == neighbour class')
            if args.pt_data == 'ori_neigh':
                return model, original_class, neighbour_class, None, None, neighbour_delta

        loss_list = []
        acc_list = []
        for _ in range(args.pt_iter):
            if args.pt_data == 'ori_neigh':
                original_data, original_label = next(iter(train_loaders_by_class[original_class]))
                neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))
            elif args.pt_data == 'ori_rand':
                original_data, original_label = next(iter(train_loaders_by_class[original_class]))
                neighbour_class = (original_class + random.randint(1, 10)) % 10
                neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))
            elif args.pt_data == 'train':
                original_data, original_label = next(iter(train_loader))
                neighbour_data, neighbour_label = next(iter(train_loader))
            else:
                raise NotImplementedError

            data = torch.vstack([original_data, neighbour_data]).to(device)
            label = torch.hstack([original_label, neighbour_label]).to(device)
            # label_mixup = torch.hstack([original_label_mixup, neighbour_label_mixup]).to(device)

            if args.pt_method == 'adv':
                # generate fgsm adv examples
                delta = (torch.rand_like(data) * 2 - 1) * epsilon  # uniform rand from [-eps, eps]
                noise_input = data + delta
                noise_input.requires_grad = True
                noise_output = model(noise_input)
                loss = loss_func(noise_output, label)  # loss to be maximized
                input_grad = torch.autograd.grad(loss, noise_input)[0]
                delta = delta + alpha * torch.sign(input_grad)
                delta.clamp_(-epsilon, epsilon)
                adv_input = data + delta
            elif args.pt_method == 'dir_adv':
                # use fixed direction attack
                if args.adv_dir == 'pos':
                    adv_input = data + 1 * neighbour_delta
                elif args.adv_dir == 'neg':
                    adv_input = data + -1 * neighbour_delta
                elif args.adv_dir == 'both':
                    directed_delta = torch.vstack([torch.ones_like(original_data).to(device) * neighbour_delta,
                                                    torch.ones_like(neighbour_data).to(device) * -1 * neighbour_delta])
                    adv_input = data + directed_delta
            elif args.pt_method == 'normal':
                adv_input = data
            else:
                raise NotImplementedError

            adv_output = model(adv_input.detach())

            loss = loss_func(adv_output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            defense_acc = cal_accuracy(adv_output, label)
            loss_list.append(loss)
            acc_list.append(defense_acc)
            # print('loss: {:.4f}  acc: {:.4f}'.format(loss, defense_acc))
    return model, original_class, neighbour_class, loss_list, acc_list, neighbour_delta


def get_args():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch-size', default=100, type=int)
    # parser.add_argument('--data-dir', default='../mnist-data', type=str)
    # parser.add_argument('--fname', type=str, default='models/fgsm.pth')
    # parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'none'])
    # parser.add_argument('--epsilon', default=0.3, type=float)
    # parser.add_argument('--attack-iters', default=50, type=int)
    # parser.add_argument('--alpha', default=1e-2, type=float)
    # parser.add_argument('--restarts', default=10, type=int)
    # parser.add_argument('--seed', default=0, type=int)
    # return parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--fname', type=str, default='models/fgsm.pth')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--pt-data', default='ori_neigh', choices=['ori_rand', 'ori_neigh', 'train'], type=str)
    parser.add_argument('--pt-method', default='adv', choices=['adv', 'dir_adv', 'normal'], type=str)
    parser.add_argument('--adv-dir', default='na', choices=['na', 'pos', 'neg', 'both'], type=str)
    parser.add_argument('--neigh-method', default='untargeted', choices=['untargeted', 'targeted'], type=str)
    parser.add_argument('--pt-iter', default=50, type=int)
    parser.add_argument('--pt-lr', default=0.001, type=float)
    parser.add_argument('--att-iter', default=40, type=int)
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
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    logger = logging.getLogger("eval")
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_train = datasets.MNIST("../mnist-data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("../mnist-data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=False)
    train_loaders_by_class = get_train_loaders_by_class("../mnist-data", batch_size=100)

    model = mnist_net().cuda()
    checkpoint = torch.load(args.fname)
    model.load_state_dict(checkpoint)
    model.eval()

    epsilon = 0.3
    alpha = 1e-2
    pgd_loss = 0
    pgd_acc = 0
    pgd_loss_post = 0
    pgd_acc_post = 0
    normal_loss = 0
    normal_acc = 0
    normal_loss_post = 0
    normal_acc_post = 0
    neighbour_acc = 0
    n = 0
    model.eval()
    cos_sim = nn.CosineSimilarity(dim=0)
    for i, (X, y) in enumerate(test_loader):
        n += y.size(0)
        X, y = X.cuda(), y.cuda()
        if not args.blackbox:
            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, args.att_iter, args.att_restart).detach()
        else:
            pgd_delta = torch.zeros_like(X)

        logger.info("\n")
        # evaluate base model
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            logger.info('Batch {}\tbase acc: {:.4f}'.format(i+1, pgd_acc / n))

        # evaluate post model against adv
        with torch.no_grad():
            post_model, original_class, neighbour_class, _, _, _ = post_train(model, X + pgd_delta, train_loader,
                                                                              train_loaders_by_class, args)
            # evaluate neighbour acc
            neighbour_acc += 1 if int(y) == int(original_class) or int(y) == int(neighbour_class) else 0
            logger.info('Batch {}\tneigh acc: {:.4f}'.format(i + 1, neighbour_acc / n))

            # evaluate prediction acc
            output = post_model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss_post += loss.item() * y.size(0)
            pgd_acc_post += (output.max(1)[1] == y).sum().item()
            logger.info('Batch {}\tadv acc (post): {:.4f}'.format(i+1, pgd_acc_post / n))

        # evaluate base model against normal
        with torch.no_grad():
            output = model(X)
            loss = F.cross_entropy(output, y)
            normal_loss += loss.item() * y.size(0)
            normal_acc += (output.max(1)[1] == y).sum().item()
            logger.info('Batch {}\tnormal acc: {:.4f}'.format(i+1, normal_acc / n))

        # evaluate post model against normal
        with torch.no_grad():
            post_model, original_class, neighbour_class, _, _, _ = post_train(model, X, train_loader,
                                                                              train_loaders_by_class, args)
            output = post_model(X)
            loss = F.cross_entropy(output, y)
            normal_loss_post += loss.item() * y.size(0)
            normal_acc_post += (output.max(1)[1] == y).sum().item()
            logger.info('Batch {}\tnormal acc (post): {:.4f}'.format(i+1, normal_acc_post / n))


if __name__ == "__main__":
    main()
