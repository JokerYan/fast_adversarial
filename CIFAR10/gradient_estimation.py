import copy
import argparse

import torch
import torch.nn as nn
import numpy as np

from post_model import PostModel
from utils import get_loaders
from boundary_attack_utils import fine_grained_binary_search


repeat_count = 5
step_size = 0.001
pixel_x = 16
pixel_y = 16
pixel_c = 0

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
    post_cos_sim_list = []
    noise_cos_sim_list = []
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        unit_error = torch.zeros_like(images)
        unit_error[0][pixel_c][pixel_x][pixel_y] = 1

        '''
        ## post model estimate
        # for j in range(repeat_count):
        #     images_pos = copy.deepcopy(images).detach() + unit_error * step_size
        #     images_neg = copy.deepcopy(images).detach() - unit_error * step_size
        #     output_pos = post_model(images_pos).detach()
        #     output_neg = post_model(images_neg).detach()
        #     loss_pos = loss_func(output_pos, labels)
        #     loss_neg = loss_func(output_neg, labels)
        #     gradient = (loss_pos - loss_neg) / (2 * step_size)
        #     print("post gradient:", float(gradient))

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
        '''

        # gradient gt post model
        all_gradient_list = []
        modified = True
        for j in range(2):
            images.requires_grad = True
            output = post_model(images, post=True)
            loss = loss_func(output, labels)
            all_gradient = torch.autograd.grad(loss, images)[0]
            all_gradient_list.append(all_gradient)
            # print("gt gradient post: {:.8f}".format(float(all_gradient[0][pixel_c][pixel_x][pixel_y])))
            # print("gt gradient post: {:.8f}".format(float(all_gradient[0][pixel_c+1][pixel_x][pixel_y])))
            # print("gt gradient post: {:.8f}".format(float(all_gradient[0][pixel_c+2][pixel_x][pixel_y])))
            if not post_model.model_modified:
                print("model not modified in post train")
                modified = False
                break
        if not modified:
            print()
            continue
        cos_sim = cos_sim_func(all_gradient_list[0].view(-1), all_gradient_list[1].view(-1))
        post_cos_sim_list.append(cos_sim)
        print("post cosine sim: ", cos_sim)

        all_gradient_list = []
        # gradient gt normal model with noise in output
        for j in range(2):
            images.requires_grad = True
            output = post_model(images, post=False)
            output_noise = torch.randn_like(output) * 0.2 + 1
            print(torch.argmax(output))
            print(torch.argmax(output * output_noise))
            loss = loss_func(output * output_noise, labels)
            all_gradient = torch.autograd.grad(loss, images)[0]
            all_gradient_list.append(all_gradient)
            # print("gt gradient: {:.8f}".format(float(all_gradient[0][pixel_c][pixel_x][pixel_y])))
            # print("gt gradient: {:.8f}".format(float(all_gradient[0][pixel_c+1][pixel_x][pixel_y])))
            # print("gt gradient: {:.8f}".format(float(all_gradient[0][pixel_c+2][pixel_x][pixel_y])))
        cos_sim = cos_sim_func(all_gradient_list[0].view(-1), all_gradient_list[1].view(-1))
        noise_cos_sim_list.append(cos_sim)
        print("noise cosine sim: ", cos_sim)

        # # boundary attack estimate
        # # theta = torch.rand_like(images)
        # theta = all_gradient.detach()
        # theta = theta / torch.linalg.norm(theta, ord=2, dim=1)
        # print(theta.shape)
        # beta = 0.005
        # u = torch.randn_like(theta)
        # g0, _ = fine_grained_binary_search(post_model, images, labels, theta)
        # g1, _ = fine_grained_binary_search(post_model, images, labels, theta + beta * u)
        # all_gradient = (g1 - g0) / beta * u
        # print("boundary gradient: {:.8f}".format(float(all_gradient[0][pixel_c][pixel_x][pixel_y])))
        # print("boundary gradient: {:.8f}".format(float(all_gradient[0][pixel_c+1][pixel_x][pixel_y])))
        # print("boundary gradient: {:.8f}".format(float(all_gradient[0][pixel_c+2][pixel_x][pixel_y])))

        print()
        if len(post_cos_sim_list) >= 20:
            break
    print("post avg:", torch.mean(torch.Tensor(post_cos_sim_list)))
    print("noise avg:", torch.mean(torch.Tensor(noise_cos_sim_list)))



if __name__ == '__main__':
    main()