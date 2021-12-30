import cv2
import torch
import numpy as np
import matplotlib
import logging

matplotlib.use('Agg')
from matplotlib import cm
from matplotlib import pyplot as plt


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def visualize_loss_surface(base_model, loss_model_list, loss_model_name_list, image, label, attack_func):
    loss_func = torch.nn.CrossEntropyLoss()
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    step_count = 10  # including origin
    pgd_delta_list = []
    for i in range(2):
        pgd_delta = attack_func(base_model, image, label, epsilon, alpha, 50, 10, opt=None)
        pgd_delta_list.append(pgd_delta)
    with torch.no_grad():
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        for model_index, loss_model in enumerate(loss_model_list):
            loss_surface = torch.zeros(step_count, step_count)
            delta_axis_x = torch.zeros(step_count)
            delta_axis_y = torch.zeros(step_count)
            for i in range(step_count):
                for j in range(step_count):
                    # delta_axis_x[i] = torch.norm(pgd_delta_list[0] * i / step_count, p=2)
                    # delta_axis_y[j] = torch.norm(pgd_delta_list[1] * j / step_count, p=2)
                    delta_axis_x[i] = i
                    delta_axis_y[j] = j
                    mix_delta = pgd_delta_list[0] * i / step_count \
                                + pgd_delta_list[1] * j / step_count
                    mix_image = image + mix_delta
                    mix_output = loss_model(mix_image)
                    mix_loss = loss_func(mix_output, label)
                    loss_surface[i][j] = mix_loss
            # print(loss_surface)
            delta_axis_x, delta_axis_y = np.meshgrid(delta_axis_x.detach().cpu().numpy(), delta_axis_y.detach().cpu().numpy())
            loss_surface = loss_surface.detach().cpu().numpy()

            surf = ax.plot_surface(delta_axis_x, delta_axis_y, loss_surface, label=loss_model_name_list[model_index])
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d
        ax.legend()
        plt.savefig('./loss_surface.png')
        print('loss surface plot saved')
        plt.close()


def visualize_decision_boundary(model, natural_input, adv_input, neighbor_input, index):
    resolution = 20

    # coordinate: (row, column)
    natural_pos = [resolution / 4, resolution / 4]
    adv_pos = [resolution * 3 / 4, resolution / 4]
    neighbor_pos = [resolution / 4, resolution * 3 / 4]

    delta1 = (adv_input - natural_input) / (adv_pos[0] - natural_pos[0])
    delta2 = (neighbor_input - natural_input) / (neighbor_pos[1] - natural_pos[1])
    pred_matrix = np.zeros([resolution, resolution])

    nat_pred = torch.argmax(model(natural_input))
    adv_pred = torch.argmax(model(adv_input))
    neigh_pred = torch.argmax(model(neighbor_input))

    for i in range(resolution):
        for j in range(resolution):
            cur_input = natural_input + (i - natural_pos[0]) * delta1 + (j - natural_pos[1]) * delta2
            cur_output = model(cur_input)
            if i == neighbor_pos[0] and j == neighbor_pos[1]:
                assert torch.argmax(cur_output) == nat_pred
            pred_matrix[i][j] = torch.argmax(cur_output)
    print(pred_matrix)
    fig, ax = plt.subplots()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    im = ax.imshow(pred_matrix)

    # add text, coordinate: (column, row)
    plt.text(natural_pos[1], natural_pos[0], 'x', fontsize=12, horizontalalignment='center',
             verticalalignment='center', c='white' if nat_pred < 5 else 'black')
    plt.text(adv_pos[1], adv_pos[0], 'x\'', fontsize=12, horizontalalignment='center',
             verticalalignment='center', c='white' if adv_pred < 5 else 'black')
    plt.text(neighbor_pos[1], neighbor_pos[0], 'x\'\'', fontsize=12, horizontalalignment='center',
             verticalalignment='center', c='white' if neigh_pred < 5 else 'black')

    plt.savefig('./debug/decision_boundary_{}.png'.format(index))
    print('decision boundary plot saved')
    plt.close()


def visualize_cam(x, cam, index):
    cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
    cifar10_std = np.array([0.2471, 0.2435, 0.2616])
    cifar10_mean = np.expand_dims(cifar10_mean, axis=(1, 2))
    cifar10_std = np.expand_dims(cifar10_std, axis=(1, 2))

    x = np.squeeze(x.cpu().numpy())
    x = 255 * (cifar10_std * x + cifar10_mean)
    x = np.transpose(x, [1, 2, 0])
    cv2.imwrite('./debug/input_{}.jpg'.format(index), x)
    fig, ax = plt.subplots()
    cam = ax.imshow(cam)
    plt.savefig('./debug/cam_{}.jpg'.format(index))


def visualize_grad(model, x, y, index):
    loss_func = torch.nn.CrossEntropyLoss()
    with torch.enable_grad():
        x.requires_grad = True
        output = model(x)
        loss = loss_func(output, y)  # loss to be maximized
        grad = torch.autograd.grad(loss, x)[0].detach().cpu().numpy()

        grad_sample = grad[0][0]
        grad_sample[grad_sample < 0] = 0  # only positive
        fig, ax = plt.subplots()
        cam = ax.imshow(grad_sample)
        plt.savefig('./debug/grad_{}.jpg'.format(index))
