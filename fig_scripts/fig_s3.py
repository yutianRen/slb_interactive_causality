"""
A 2-d example to visualize the input-output relation,
as shown in supplementary Section 3.

"""

import bisect

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve


def get_x_initial(x1=(0.1, 10), x2=(0.1, 10)):
    """
    get the initial values of vector x

    """
    x1 = np.linspace(x1[0], x1[1], 50)
    x2 = np.linspace(x2[0], x2[1], 50)

    rt = np.dstack(np.meshgrid(x1, x2)).reshape(-1, 2)
    return rt


def unpert_get_t(x_initial, x1_end) -> np.ndarray:
    """
    use x initial and the final value of x1 to derive the time interval in unperturbed case.
    """

    size = x_initial.shape[0]
    rt = np.zeros(size)

    for i in range(size):
        a, b = x_initial[i]
        rt[i] = 1/3 * np.log((3*x1_end-2*a+2*b)/(a+2*b))
    return rt

def unpert_get_y_end(t_arr, x_initial) -> np.ndarray:
    """
    to derive the final values of vector y given the interaction time and initial values of x

    t_arr: the interaction time.
    x_initial: initial values of vector x
    """
    size = len(t_arr)
    rt = np.zeros((size, 2))

    for i in range(size):
        a, b = x_initial[i]
        t = t_arr[i]

        part1 = -a + b + np.exp(2*t)*(12-3*a-9*b) + np.exp(3*t)*(4*a+8*b)
        rt[i][0] = 1/12 * (part1 + 6 * a * t - 6 * b * t)
        rt[i][1] = 1/12 * (part1 - 6 * a * t + 6 * b * t)
    return rt

def pert_get_t(x_initial, x1_end) -> np.ndarray:
    """
    under perturbation to get t_true

    """
    size = x_initial.shape[0]
    rt = np.zeros(size)

    for i in range(size):
        a, b = x_initial[i]
        rt[i] = 1/3 * np.log((3*x1_end+1-2*a+2*b)/(1+a+2*b))
    return rt


def pert_get_y_end(t_arr, x_initial) -> np.ndarray:
    """
    under perturbation to get the final values of vector y.

    """
    size = len(t_arr)
    rt = np.zeros((size, 2))

    for i in range(size):
        a, b = x_initial[i]
        t = t_arr[i]

        part1 = 2 - a + b + np.exp(2*t)*(6-3*a-9*b) + np.exp(3*t)*(4+4*a+8*b)

        rt[i][0] = 1/12 * (part1 + 6 * a * t - 6 * b * t)
        rt[i][1] = 1/12 * (part1 - 6 * a * t + 6 * b * t)
    return rt

def pert_x_propagate(x_initial, t_arr):
    """
    defines the evolution function of x under perturbation

    """

    size = len(t_arr)
    out = np.zeros((size, 2))

    for i in range(size):
        t = t_arr[i]
        a, b = x_initial[i]
        out[i][0] = 1/3*(-1 + 2*a - 2*b + np.exp(3*t)*(1 + a + 2*b))
        out[i][1] = 1/3*(-1 - a + b + np.exp(3*t)*(1 + a + 2*b))
    return out

def pert_get_xslb_tif(x_initial, x1_end=10):
    """
    get the inferred interaction time (tif) under perturbation

    """
    size = len(x_initial)
    out_tif = np.zeros(size)

    for i in range(size):
        a, b = x_initial[i]
        x1_slb = a
        out_tif[i] = 1/3 * np.log((3*x1_end+1-2*a+2*b) / (3*x1_slb+1-2*a+2*b))

    return out_tif

def unpert_convert_AB_to_xend(tif, x_initial, x1_end=10):
    """


    """

    size = len(tif)
    out = np.zeros((size, 2))

    for i in range(size):
        t = tif[i]
        a, b = x_initial[i]
        x2_end = - a + b + x1_end
        out[i][0] = 1/3 * (np.exp(-3*t) * (x1_end+2*x2_end) + 2*x1_end - 2*x2_end)
        out[i][1] = 1/3 * (np.exp(-3*t) * (x1_end+2*x2_end) - x1_end + x2_end)

    return out

def unpert_get_y_given_tif(tif, AB):
    size = len(tif)
    out = np.zeros((size, 2))

    for i in range(size):
        a, b = AB[i]
        t = tif[i]
        part1 = -a + b + np.exp(2*t)*(12-3*a-9*b) + np.exp(3*t)*(4*a+8*b)
        # print(a, b)
        out[i][0] = 1/12 * (part1 + 6*a*t-6*b*t)
        out[i][1] = 1/12 * (part1 - 6*a*t + 6*b*t)
    return out


def get_slb_x(x_initial, t_infer, t_true, x1_end=10):
    """
    derive the self-labeled x given t_infer under perturbation.

    """
    t_diff = t_true - t_infer
    slb_x = pert_x_propagate(x_initial, t_diff)

    return slb_x

def get_t_infer(y1f_arr, y2f_arr, x1f):
    """
    use numerical method to solve a complex function to derive t_infer.

    """

    t_rt = []
    for y1f, y2f in zip(y1f_arr, y2f_arr):
        m = y1f - y2f
        def func(t):
            return 1/6*(12*x1f -9*m/t + 12 * np.exp(2*t) - 4 * (3*x1f-2*m/t) * np.exp(-t) + m/t*np.exp(2*t)) - y1f - y2f
        t_cand_arr = np.linspace(1e-9, 15, 10000)
        f_arr = func(t_cand_arr)

        idx = bisect.bisect_left(f_arr, 0)
        t_search_start = t_cand_arr[idx]

        t_root = fsolve(func, t_search_start, xtol=1e-3, maxfev=1000000)
        t_rt.append(t_root[0])

    return np.array(t_rt)



if __name__ == '__main__':

    x_initial = get_x_initial()
    t_unpert = unpert_get_t(x_initial, 10)
    y_end_trad = unpert_get_y_end(t_unpert, x_initial)

    t_true = pert_get_t(x_initial, 10)
    y_end_gt = pert_get_y_end(t_true, x_initial)

    t_infer = get_t_infer(y_end_gt[:, 0], y_end_gt[:, 1], 10)


    x_slb = get_slb_x(x_initial, t_infer, t_true, 10)

    # figure
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    fig = plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98, wspace=0.15, hspace=0.40)

    # let x_initial = x_slb
    # then recalculate y_gt and y_trad for plots.
    x_initial = x_slb
    t_unpert = unpert_get_t(x_initial, 10)
    y_end_trad_use_slb = unpert_get_y_end(t_unpert, x_initial)
    t_true = pert_get_t(x_initial, 10)
    y_end_gt_use_slb = pert_get_y_end(t_true, x_initial)

    offset = 10
    y_end_trad_use_slb += offset
    y_end_gt_use_slb -= offset


    ydim = 0
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter3D(x_initial[:, 0], x_initial[:, 1], y_end_trad_use_slb[:, ydim], label='trad')
    ax1.scatter3D(x_initial[:, 0], x_initial[:, 1], y_end_gt_use_slb[:, ydim], label='fs')
    ax1.scatter3D(x_slb[:, 0], x_slb[:, 1], y_end_gt[:, ydim], label='slb')
    ax1.legend(fontsize=20)

    ax1.set_xlabel(r'$x_{1}$', fontsize=22)
    ax1.set_ylabel(r'$x_{2}$', fontsize=22)
    ax1.set_zlabel(r'$y_{1}$', fontsize=22)
    ax1.view_init(elev=5., azim=-42)

    ydim = 1
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter3D(x_initial[:, 0], x_initial[:, 1], y_end_trad_use_slb[:, ydim], label='trad')
    ax2.scatter3D(x_initial[:, 0], x_initial[:, 1], y_end_gt_use_slb[:, ydim], label='fs')
    ax2.scatter3D(x_slb[:, 0], x_slb[:, 1], y_end_gt[:, ydim], label='slb')
    ax2.legend(fontsize=20)

    ax2.set_xlabel(r'$x_{1}$', fontsize=22)
    ax2.set_ylabel(r'$x_{2}$', fontsize=22)
    ax2.set_zlabel(r'$y_{2}$', fontsize=22)
    ax2.view_init(elev=5., azim=-42)



    plt.legend(loc="best", fontsize=20)
    # plt.savefig('2d_example.png', dpi=300, bbox_inches='tight',pad_inches = 0.05)
    plt.show()
