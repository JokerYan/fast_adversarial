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


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
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


def post_train(model, images, train_loaders_by_class):
    alpha = 10 / 255
    epsilon = 8 / 255
    loss_func = nn.CrossEntropyLoss()
    device = torch.device('cuda')
    model = copy.deepcopy(model)
    # model.train()
    fix_model = copy.deepcopy(model)
    # attack_model = torchattacks.PGD(model, eps=(8/255)/std, alpha=(2/255)/std, steps=20)
    optimizer = torch.optim.SGD(lr=0.001,
                                params=model.parameters(),
                                momentum=0.9,
                                nesterov=True)
    # target_bce_loss_func = TargetBCELoss()
    # target_bl_loss_func = TargetBLLoss()
    with torch.enable_grad():
        # find neighbour
        original_output = fix_model(images)
        original_class = torch.argmax(original_output).reshape(1)
        # neighbour_images = attack_model(images, original_class)
        neighbour_images = attack_pgd(model, images, original_class, epsilon, alpha, attack_iters=50, restarts=10) + images
        neighbour_output = fix_model(neighbour_images)
        neighbour_class = torch.argmax(neighbour_output).reshape(1)

        if original_class == neighbour_class:
            print('original class == neighbour class')
            # return model, original_class, neighbour_class, None, None
            neighbour_lost = True

        loss_list = []
        acc_list = []
        for _ in range(50):
            neighbour_class = (original_class + random.randint(1, 9)) % 10
            original_data, original_label = next(iter(train_loaders_by_class[original_class]))
            neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))

            data = torch.vstack([original_data, neighbour_data]).to(device)
            data = merge_images(data, images, 0.7, device)
            label = torch.hstack([original_label, neighbour_label]).to(device)
            target = torch.hstack([neighbour_label, original_label]).to(device)

            # generate fgsm adv examples
            delta = (torch.rand_like(data) * 2 - 1) * epsilon  # uniform rand from [-eps, eps]
            noise_input = data + delta
            noise_input.requires_grad = True
            noise_output = model(noise_input)
            loss = loss_func(noise_output, label)  # loss to be maximized
            # loss = target_bce_loss_func(noise_output, label, original_class, neighbour_class)  # bce loss to be maximized
            input_grad = torch.autograd.grad(loss, noise_input)[0]
            delta = delta + alpha * torch.sign(input_grad)
            delta.clamp_(-epsilon, epsilon)
            adv_input = data + delta

            # generate pgd adv example
            # attack_model.set_mode_targeted_by_function(lambda im, la: target)
            # adv_input = attack_model(data, label)

            adv_output = model(adv_input.detach())
            # adv_class = torch.argmax(adv_output)
            loss_pos = loss_func(adv_output, label)
            loss_neg = loss_func(adv_output, target)
            # bce_loss = target_bce_loss_func(adv_output, label, original_class, neighbour_class)
            # bl_loss = target_bl_loss_func(adv_output, label, original_class, neighbour_class)

            # loss = torch.mean(loss_list)
            loss = loss_pos
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            defense_acc = cal_accuracy(adv_output, label)
            loss_list.append(loss)
            acc_list.append(defense_acc)
            # print('loss: {:.4f}  acc: {:.4f}'.format(loss, defense_acc))
    return model, original_class, neighbour_class, loss_list, acc_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    parser.add_argument('--fname', type=str)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'none'])
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--attack-iters', default=50, type=int)
    parser.add_argument('--alpha', default=1e-2, type=float)
    parser.add_argument('--restarts', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_test = datasets.MNIST("../mnist-data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=False)
    train_loaders_by_class = get_train_loaders_by_class("../mnist-data", batch_size=100)

    model = mnist_net().cuda()
    checkpoint = torch.load(args.fname)
    model.load_state_dict(checkpoint)
    model.eval()

    # total_loss = 0
    # total_acc = 0
    # n = 0
    #
    # if args.attack == 'none':
    #     with torch.no_grad():
    #         for i, (X, y) in enumerate(test_loader):
    #             X, y = X.cuda(), y.cuda()
    #             output = model(X)
    #             loss = F.cross_entropy(output, y)
    #             total_loss += loss.item() * y.size(0)
    #             total_acc += (output.max(1)[1] == y).sum().item()
    #             n += y.size(0)
    # else:
    #     for i, (X, y) in enumerate(test_loader):
    #         X, y = X.cuda(), y.cuda()
    #         if args.attack == 'pgd':
    #             delta = attack_pgd(model, X, y, args.epsilon, args.alpha, args.attack_iters, args.restarts)
    #         elif args.attack == 'fgsm':
    #             delta = attack_fgsm(model, X, y, args.epsilon)
    #         with torch.no_grad():
    #             output = model(X + delta)
    #             loss = F.cross_entropy(output, y)
    #             total_loss += loss.item() * y.size(0)
    #             total_acc += (output.max(1)[1] == y).sum().item()
    #             n += y.size(0)

    epsilon = args.epsilon
    alpha = args.alpha
    pgd_loss = 0
    pgd_acc = 0
    pgd_loss_post = 0
    pgd_acc_post = 0
    normal_loss_post = 0
    normal_acc_post = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        n += y.size(0)
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, args.attack_iters, args.restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            print('Batch {}  avg acc: {}'.format(i, pgd_acc / n))
        post_model, _, _, _, _ = post_train(model, X, train_loaders_by_class)
        with torch.no_grad():
            output = post_model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss_post += loss.item() * y.size(0)
            pgd_acc_post += (output.max(1)[1] == y).sum().item()
            print('Batch {}  avg post acc: {}'.format(i, pgd_acc_post / n))
        with torch.no_grad():
            output = post_model(X)
            loss = F.cross_entropy(output, y)
            normal_loss_post += loss.item() * y.size(0)
            normal_acc_post += (output.max(1)[1] == y).sum().item()
            print('Batch {}  normal post acc: {}'.format(i, normal_acc_post / n))
        print()

    logger.info('Normal Loss \t Normal Acc \t PGD Loss \t PGD Acc \t PGD Post Loss \t PGD Post Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f', normal_loss_post/n, normal_acc_post/n, pgd_loss/n, pgd_acc/n, pgd_loss_post/n, pgd_acc_post/n)


if __name__ == "__main__":
    main()
