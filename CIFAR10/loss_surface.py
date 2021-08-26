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


def calculate_loss_surface(base_model, loss_model_list, loss_model_name_list, image, label, attack_func):
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
        for loss_model in loss_model_list:
            loss_surface = torch.zeros(step_count, step_count)
            delta_axis_x = torch.zeros(step_count)
            delta_axis_y = torch.zeros(step_count)
            for i in range(step_count):
                for j in range(step_count):
                    delta_axis_x[i] = torch.norm(pgd_delta_list[0] * i / step_count, p=2)
                    delta_axis_y[j] = torch.norm(pgd_delta_list[1] * j / step_count, p=2)
                    mix_delta = pgd_delta_list[0] * i / step_count \
                                + pgd_delta_list[1] * j / step_count
                    mix_image = image + mix_delta
                    mix_output = loss_model(mix_image)
                    mix_loss = loss_func(mix_output, label)
                    loss_surface[i][j] = mix_loss
            print(loss_surface)
            delta_axis_x = np.meshgrid(delta_axis_x.detach().cpu().numpy())
            delta_axis_y = np.meshgrid(delta_axis_y.detach().cpu().numpy())
            loss_surface = loss_surface.detach().cpu().numpy()

            ax.plot_surface(delta_axis_x, delta_axis_y, loss_surface, linewidth=0, cmap=cm.coolwarm)
        plt.savefig('./loss_surface.png')
        input('loss surface plot saved')
