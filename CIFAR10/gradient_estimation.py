import random
import copy
import argparse

import torch
import torch.nn as nn
import numpy as np

from post_model import PostModel
from utils import get_loaders
from boundary_attack_utils import fine_grained_binary_search


step_size = 0.001
# pixel_x = 16
# pixel_y = 16
# pixel_c = 0
noise_ratio = 0.4

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.set_defaults(mixup=True, type=bool)
    parser.add_argument('--no-mixup', dest='mixup', action='store_false')
    parser.add_argument('--pt-data', default='ori_neigh', choices=['ori_rand', 'ori_train', 'ori_neigh_train', 'ori_neigh', 'rand'], type=str)
    parser.add_argument('--pt-method', default='adv', choices=['adv', 'normal'], type=str)
    parser.add_argument('--pt-iter', default=5, type=int)
    parser.set_defaults(rs_neigh=True, type=bool)
    parser.add_argument('--no-rs-neigh', dest='rs_neigh', action='store_false')
    parser.set_defaults(blackbox=False, type=bool)
    parser.add_argument('--blackbox', dest='blackbox', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    post_model = PostModel(model=None, args=args)
    _, test_loader = get_loaders(args.data_dir, batch_size=1)
    loss_func = nn.CrossEntropyLoss()
    cos_sim_func = nn.CosineSimilarity(dim=0)

    post_acc_list = []
    noise_acc_list = []
    normal_acc_list = []
    post_cos_sim_list = []
    noise_cos_sim_list = []
    post_same_dir_ratio_list = []
    post_same_dir_boundary_ratio_list = []
    post_same_dir_estimate_ratio_list = []
    for i, (images, labels) in enumerate(test_loader):
        # print(len(post_cos_sim_list))
        print(len(post_same_dir_boundary_ratio_list))
        images = images.cuda()
        labels = labels.cuda()

        ## normal model estimate
        # for j in range(repeat_count):
        #     images_pos = copy.deepcopy(images).detach() + unit_error * step_size
        #     images_neg = copy.deepcopy(images).detach() - unit_error * step_size
        #     output_pos = post_model(images_pos, post=False).detach()
        #     output_neg = post_model(images_neg, post=False).detach()
        #     loss_pos = loss_func(output_pos, labels)
        #     loss_neg = loss_func(output_neg, labels)
        #     gradient = (loss_pos - loss_neg) / (2 * step_size)
        #     print("normal gradient:", float(gradient))

        ## normal model with output noise estimate
        # for j in range(repeat_count):
        #     images_pos = copy.deepcopy(images).detach() + unit_error * step_size
        #     images_neg = copy.deepcopy(images).detach() - unit_error * step_size
        #     output_pos = post_model(images_pos, post=False).detach()
        #     output_neg = post_model(images_neg, post=False).detach()
        #
        #     sum_output_pos = torch.zeros_like(output_pos)
        #     sum_output_neg = torch.zeros_like(output_neg)
        #     average_count = 10000
        #     for k in range(average_count):
        #         # add noise
        #         output_pos_noise = torch.randn_like(output_pos) * 0.03 + 1
        #         output_neg_noise = torch.randn_like(output_neg) * 0.03 + 1
        #         sum_output_pos += output_pos * output_pos_noise
        #         sum_output_neg += output_neg * output_neg_noise
        #     output_pos = sum_output_pos / average_count
        #     output_neg = sum_output_neg / average_count
        #
        #     loss_pos = loss_func(output_pos, labels)
        #     loss_neg = loss_func(output_neg, labels)
        #     gradient = (loss_pos - loss_neg) / (2 * step_size)
        #     print("normal gradient:", float(gradient))
        # for j in range(repeat_count):
        #     images_pos = copy.deepcopy(images).detach() + unit_error * step_size
        #     images_neg = copy.deepcopy(images).detach() - unit_error * step_size
        #     output_pos = post_model(images_pos, post=False).detach()
        #     output_neg = post_model(images_neg, post=False).detach()
        #     loss_pos = loss_func(output_pos, labels)
        #     loss_neg = loss_func(output_neg, labels)
        #     gradient = (loss_pos - loss_neg) / (2 * step_size)
        #     print("normal gradient:", float(gradient))

        # normal
        output = post_model(images, post=False)
        acc = 1 if torch.argmax(output) == labels else 0
        normal_acc_list.append(acc)

        # # gradient gt post model
        # all_gradient_list = []
        # modified = True
        # for j in range(2):
        #     images.requires_grad = True
        #     output = post_model(images, post=True)
        #     loss = loss_func(output, labels)
        #     all_gradient = torch.autograd.grad(loss, images)[0]
        #     all_gradient_list.append(all_gradient)
        #     if not post_model.model_modified:
        #         print("model not modified in post train")
        #         modified = False
        #         break
        #     acc = 1 if torch.argmax(output) == labels else 0
        #     post_acc_list.append(acc)
        # if not modified:
        #     print()
        #     continue
        # # print(all_gradient_list[0][0][0][16][16])
        # # print(all_gradient_list[1][0][0][16][16])
        # gradient_direction = all_gradient_list[0] * all_gradient_list[1]
        # gradient_same_dir_ratio = torch.mean(torch.where(gradient_direction > 0, torch.ones_like(gradient_direction), torch.zeros_like(gradient_direction)))
        # post_same_dir_ratio_list.append(gradient_same_dir_ratio)
        # print("post gradient same ratio:", gradient_same_dir_ratio)
        # cos_sim = cos_sim_func(all_gradient_list[0].view(-1), all_gradient_list[1].view(-1))
        # post_cos_sim_list.append(cos_sim)
        # print("post cosine sim: ", cos_sim)

        # all_gradient_list = []
        # # gradient gt normal model with noise in output
        # for j in range(2):
        #     images.requires_grad = True
        #     output = post_model(images, post=False)
        #     output_noise = torch.rand_like(output) * noise_ratio - noise_ratio / 2 + 1
        #     # print(output_noise)
        #     # print(torch.argmax(output))
        #     # print(torch.argmax(output * output_noise))
        #     output = output * output_noise
        #     loss = loss_func(output, labels)
        #     all_gradient = torch.autograd.grad(loss, images)[0]
        #     all_gradient_list.append(all_gradient)
        #
        #     acc = 1 if torch.argmax(output) == labels else 0
        #     noise_acc_list.append(acc)
        #
        # gradient_direction = all_gradient_list[0] * all_gradient_list[1]
        # gradient_same_dir_ratio = torch.mean(torch.where(gradient_direction > 0, torch.ones_like(gradient_direction), torch.zeros_like(gradient_direction)))
        # print("noise gradient same ratio:", gradient_same_dir_ratio)
        # cos_sim = cos_sim_func(all_gradient_list[0].view(-1), all_gradient_list[1].view(-1))
        # noise_cos_sim_list.append(cos_sim)
        # print("noise cosine sim: ", cos_sim)

        # boundary attack estimate
        images.requires_grad = True
        output = post_model(images, post=True)
        if not post_model.model_modified:
            print("model not modified in post train")
            continue
        loss = loss_func(output, labels)
        all_gradient = torch.autograd.grad(loss, images)[0]

        # theta = torch.rand_like(images)
        theta = all_gradient.detach()
        theta = theta / torch.linalg.norm(theta, ord=2, dim=1)
        print(theta.shape)
        beta = 0.005
        u = torch.randn_like(theta)
        all_gradient_list = []
        boundary_found = True
        for j in range(2):
            post_model_fix = post_model.get_post_model(images)
            g0, _ = fine_grained_binary_search(post_model_fix, images, labels, theta)
            post_model_fix = post_model.get_post_model(images)
            g1, _ = fine_grained_binary_search(post_model_fix, images, labels, theta + beta * u)
            if g0 is None or g1 is None:
                print("boundary gradient estimation failed, skipping")
                boundary_found = False
                break
            all_gradient = (g1 - g0) / beta * u
            all_gradient_list.append(all_gradient)
            # print("boundary gradient: {:.8f}".format(float(all_gradient[0][pixel_c][pixel_x][pixel_y])))
            # print("boundary gradient: {:.8f}".format(float(all_gradient[0][pixel_c+1][pixel_x][pixel_y])))
            # print("boundary gradient: {:.8f}".format(float(all_gradient[0][pixel_c+2][pixel_x][pixel_y])))
            print(g1, g0)
        if not boundary_found:
            continue
        gradient_direction = all_gradient_list[0] * all_gradient_list[1]
        gradient_same_dir_ratio = torch.mean(torch.where(gradient_direction > 0, torch.ones_like(gradient_direction), torch.zeros_like(gradient_direction)))
        post_same_dir_boundary_ratio_list.append(gradient_same_dir_ratio)
        print("post gradient boundary same ratio:", gradient_same_dir_ratio)

        # # post model estimate
        # unit_error = torch.zeros_like(images)
        # pixel_c = random.randint(0, 2)
        # pixel_x = random.randint(0, 31)
        # pixel_y = random.randint(0, 31)
        # unit_error[0][pixel_c][pixel_x][pixel_y] = 1
        # gradient_list = []
        # for j in range(2):
        #     images_pos = copy.deepcopy(images).detach() + unit_error * step_size
        #     images_neg = copy.deepcopy(images).detach() - unit_error * step_size
        #     output_pos = post_model(images_pos, post=True).detach()
        #     output_neg = post_model(images_neg, post=True).detach()
        #     loss_pos = loss_func(output_pos, labels)
        #     loss_neg = loss_func(output_neg, labels)
        #     gradient = (loss_pos - loss_neg) / (2 * step_size)
        #     gradient_list.append(gradient)
        #     print("post gradient estimate:", float(gradient))
        # gradient_same_dir_ratio = 1 if gradient_list[0] * gradient_list[1] > 0 else 0
        # post_same_dir_estimate_ratio_list.append(gradient_same_dir_ratio)
        # print("post gradient estimate same ratio:", gradient_same_dir_ratio)

        print("post cos sim avg:", torch.mean(torch.Tensor(post_cos_sim_list)))
        print("noise cos sim avg:", torch.mean(torch.Tensor(noise_cos_sim_list)))
        # print("normal acc avg:", torch.mean(torch.Tensor(normal_acc_list)))
        # print("post acc avg:", torch.mean(torch.Tensor(post_acc_list)))
        # print("noise acc avg:", torch.mean(torch.Tensor(noise_acc_list)))
        # print("post same dir ratio:", torch.mean(torch.Tensor(post_same_dir_ratio_list)))
        print("post same dir boundary ratio:", torch.mean(torch.Tensor(post_same_dir_boundary_ratio_list)))
        # print("post same dir estimate ratio:", torch.mean(torch.Tensor(post_same_dir_estimate_ratio_list)))
        print()

        # if len(post_same_dir_ratio_list) >= 100:
        #     break
        if len(post_same_dir_boundary_ratio_list) >= 100:
            break


if __name__ == '__main__':
    main()