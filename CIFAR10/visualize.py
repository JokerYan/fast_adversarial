import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import cm
from matplotlib import pyplot as plt


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()


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


def visualize_decision_boundary(model, natural_input, adv_input, neighbor_input):
    resolution = 20

    natural_pos = [resolution / 4, resolution / 4]
    adv_pos = [resolution * 3 / 4, resolution / 4]
    neighbour_pos = [resolution / 4, resolution * 3 / 4]

    delta1 = (adv_input - natural_input) / (adv_pos[0] - natural_pos[0])
    delta2 = neighbor_input - natural_input / (neighbour_pos[1] - natural_pos[1])
    pred_matrix = np.zeros([resolution, resolution])

    for i in range(resolution):
        for j in range(resolution):
            cur_input = natural_input + (i - natural_pos[0]) * delta1 + (j - natural_pos[1]) * delta2
            cur_output = model(cur_input)
            pred_matrix[i][j] = torch.argmax(cur_output)
    print(pred_matrix)
    fig, ax = plt.subplots()
    im = ax.imshow(pred_matrix)

    # add text
    plt.text(natural_pos[0], natural_pos[1], 'x', fontsize=12, horizontalalignment='center',
             verticalalignment='center', c='white')
    plt.text(adv_pos[0], adv_pos[1], 'x\'', fontsize=12, horizontalalignment='center',
             verticalalignment='center', c='white')
    plt.text(neighbour_pos[0], neighbour_pos[1], 'x\'\'', fontsize=12, horizontalalignment='center',
             verticalalignment='center', c='white')

    plt.savefig('./decision_boundary.png')
    print('decision boundary plot saved')
    plt.close()

