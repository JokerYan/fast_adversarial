# adapted from https://github.com/LeMinhThong/blackbox-attack/blob/master/boundary_attack.py

import torch
import numpy as np


def fine_grained_binary_search(model, x0, y0, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd
    while torch.argmax(model(x0 + lbd*theta)) == y0:
        lbd *= 2.0
        nquery += 1

    num_intervals = 100

    lambdas = np.linspace(0.0, lbd, num_intervals)[1:]
    lambdas = torch.from_numpy(lambdas)
    lbd_hi = lbd
    lbd_hi_index = 0
    for i, lbd in enumerate(lambdas):
        print(nquery)
        nquery += 1
        if torch.argmax(model(x0 + lbd*theta)) != y0:
            lbd_hi = lbd
            lbd_hi_index = i
            break

    lbd_lo = lambdas[lbd_hi_index - 1]

    while (lbd_hi - lbd_lo) > 1e-7:
        print("binary", nquery, lbd_hi, lbd_lo)
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if torch.argmax(model(x0 + lbd_mid*theta)) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery