import torch


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()


def calculate_loss_surface(base_model, loss_model_list, image, label, attack_func):
    loss_func = torch.nn.CrossEntropyLoss()
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    step_count = 10  # including origin
    pgd_delta_list = []
    for i in range(2):
        pgd_delta = attack_func(base_model, image, label, epsilon, alpha, 50, 10, opt=None)
        pgd_delta_list.append(pgd_delta)
    with torch.no_grad():
        for loss_model in loss_model_list:
            loss_surface = torch.zeros(step_count, step_count)
            for i in range(step_count):
                for j in range(step_count):
                    mix_delta = pgd_delta_list[0] * i / step_count \
                                + pgd_delta_list[1] * j / step_count
                    mix_image = image + mix_delta
                    mix_output = loss_model(mix_image)
                    mix_loss = loss_func(mix_output, label)
                    loss_surface[i][j] = mix_loss
            print(loss_surface)