import copy
import random

import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from tqdm import tqdm
import torchattacks

from loss_surface import calculate_loss_surface

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    return train_loader, test_loader


def get_train_loaders_by_class(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    train_dataset = datasets.CIFAR10(
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


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None, random_start=True):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if random_start:
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            print('Batch {}  avg acc: {}'.format(i, pgd_acc / n))
    return pgd_loss/n, pgd_acc/n


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


def post_train(model, images, train_loader, train_loaders_by_class, args):
    alpha = (10 / 255) / std
    epsilon = (8 / 255) / std
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
    images = images.detach()
    with torch.enable_grad():
        # find neighbour
        original_output = fix_model(images)
        original_class = torch.argmax(original_output).reshape(1)

        # neighbour_images = attack_model(images, original_class)
        rs_neighbour = args.rs_neigh
        neighbour_images = attack_pgd(model, images, original_class, epsilon, alpha, attack_iters=20, restarts=1,
                                      random_start=rs_neighbour) + images
        neighbour_output = fix_model(neighbour_images)
        neighbour_class = torch.argmax(neighbour_output).reshape(1)

        if original_class == neighbour_class:
            print('original class == neighbour class')
            # return model, original_class, neighbour_class, None, None

        loss_list = []
        acc_list = []
        # original_class = (original_class + random.randint(0, 10)) % 10
        for _ in range(args.pt_iter):
            # # randomize neighbour
            # if args.pt_data == 'ori_rand':
            #     neighbour_class = (original_class + random.randint(1, 10)) % 10
            # elif args.pt_data == 'rand':
            #     original_class = (original_class + random.randint(0, 10)) % 10
            #     neighbour_class = (original_class + random.randint(0, 10)) % 10
            # elif args.pt_data == 'ori_train':
            #     pass
            # else:
            #     raise NotImplementedError

            original_data, original_label = next(iter(train_loaders_by_class[original_class]))
            train_data, train_label = next(iter(train_loader))
            if args.pt_data == 'ori_train':
                neighbour_data, neighbour_label = next(iter(train_loader))
            else:
                neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))

            if args.pt_data == 'ori_neigh_train':
                data = torch.vstack([original_data, neighbour_data, train_data]).to(device)
                label = torch.hstack([original_label, neighbour_label, train_label]).to(device)
            else:
                data = torch.vstack([original_data, neighbour_data]).to(device)
                label = torch.hstack([original_label, neighbour_label]).to(device)

            if args.mixup:
                data = merge_images(data, images, 0.7, device)
            # target = torch.hstack([neighbour_label, original_label]).to(device)

            # data, label = next(iter(train_loader))
            # data = data.to(device)
            # label = label.to(device)

            # generate fgsm adv examplesp
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

            if args.pt_method == 'adv':
                adv_output = model(adv_input.detach())
            elif args.pt_method == 'normal':
                adv_output = model(data.detach())  # non adv training
            else:
                raise NotImplementedError
            # adv_class = torch.argmax(adv_output)
            loss_pos = loss_func(adv_output, label)
            # loss_neg = loss_func(adv_output, target)
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


def evaluate_pgd_post(test_loader, train_loader, train_loaders_by_class, model, attack_iters, restarts, args):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    pgd_loss_post = 0
    pgd_acc_post = 0
    normal_loss_post = 0
    normal_acc_post = 0
    double_attack_loss = 0
    double_attack_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        n += y.size(0)
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts).detach()
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            pgd_output_class = torch.argmax(output)
            # print("adv class: {} adv output: {}".format(pgd_output_class, output))
            print('Batch {}  avg acc: {}'.format(i, pgd_acc / n))
        post_model, _, _, _, _ = post_train(model, X + pgd_delta, train_loader, train_loaders_by_class, args)
        with torch.no_grad():
            output = post_model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss_post += loss.item() * y.size(0)
            pgd_acc_post += (output.max(1)[1] == y).sum().item()
            pgd_output_class_post = torch.argmax(output)
            # print("post class: {} post output: {}".format(pgd_output_class_post, output))
            print('Batch {}  avg post acc: {}'.format(i, pgd_acc_post / n))
        with torch.no_grad():
            output = post_model(X)
            loss = F.cross_entropy(output, y)
            normal_loss_post += loss.item() * y.size(0)
            normal_acc_post += (output.max(1)[1] == y).sum().item()
            normal_output_class_post = torch.argmax(output)
            print('Batch {}  normal post acc: {}'.format(i, normal_acc_post / n))
        pgd_delta = attack_pgd(post_model, X, y, epsilon, alpha, attack_iters, restarts).detach()
        with torch.no_grad():
            output = post_model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            double_attack_loss += loss.item() * y.size(0)
            double_attack_acc += (output.max(1)[1] == y).sum().item()
            print('Batch {}  avg double attack acc: {}'.format(i, double_attack_acc / n))
        # calculate_loss_surface(model, [model, post_model], ['model', 'post_model'], X, y, attack_func=attack_pgd)
        print('label: {}  pgd: {}  pgd_post: {}  normal_post: {}'.format(int(y), int(pgd_output_class), int(pgd_output_class_post), int(normal_output_class_post)))
        print()

    return pgd_loss/n, pgd_acc/n, pgd_loss_post/n, pgd_acc_post/n, normal_loss_post/n, normal_acc_post/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n
