import matplotlib
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.rcParams['agg.path.chunksize'] = 10000


if __name__ == '__main__':

    x2 = 100 #100
    y1 = 10
    
    x1 = np.linspace(0.1, x2, 2000)

    # for dx
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)
    fig = plt.figure(figsize=(13, 10), facecolor='w', edgecolor='k') # for dx
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98, wspace=0.25, hspace=0.40)


    error = 0.6
    # DS:
    # no disturbance: x dot = x, y dot = y+x
    # with disturbance: x dot = x+x, y dot = y+x
    if x2 > 0: # for positive systems
        y2_slb = x2*np.log(np.sqrt(x2/x1)) + y1*np.sqrt(x2/x1) # checked
        # y2_slb_e = (x2*np.log((x2/x1)**(error/2)) + y1*(x2/x1)**(error/2)) * 1
        y2_forw = x2/x1 * (x2-x1+y1) # checked
        y2_trad = x2*np.log(x2/x1) + x2/x1 * y1 # checked
        # y2_gt = x2 - np.sqrt(x1*x2) + y1*np.sqrt(x2/x1)
        y2_gt = x2 + (y1-x1) * np.sqrt(x2/x1) # checked
    else: # for negative systems
        print('negative system')
        y2_slb = np.sqrt(x1 / x2) * (0.5 * np.sqrt(x1 * x2) * (x2 / x1 - 1) + y1)
        y2_forw = x1 / x2 * (1/3 * np.abs(x1) * (np.abs(x1/x2)**(-3) - 1) + y1)
        y2_trad = x1 / x2 * (0.5 * np.abs(x1) * ((x2 / x1)**2 - 1) + y1)
        y2_gt  = np.sqrt(x1 / x2) * (1/3 * np.abs(x1) * (np.abs(x1/x2)**(-3/2) - 1) + y1)


    ax1 = plt.subplot(221)
    ax1.set_yscale('symlog')
    ax1.set_title(r'$f(x)=x, d(x)=x$', fontsize=24)
    ax1.plot(x1, y2_trad, label='trad', linewidth=2.5)
    ax1.plot(x1, y2_gt, label='fs', linewidth=2.5)
    ax1.plot(x1, y2_slb, label='slb', linewidth=2.5)
    ax1.plot(x1, y2_forw, label='fwd', linewidth=2.5)
    ax1.set_xlabel(r'$x_{1}$', fontsize=24)
    ax1.set_ylabel(r'$y_{2}$', fontsize=24)
    ax1.legend(fontsize=20)

    # DS:
    # no disturbance: x dot = x, y dot = y+x
    # with disturbance: x dot = x - 1/2*x, y dot = y + x
    # Only for positive systems.
    y2_slb = x2*np.log(x2**2/np.square(x1)) + y1*x2**2/np.square(x1) # checked
    # y2_slb_e = x2*2*error*np.log(x2/x1) + y1*(x2/x1)**(2*error) # checked

    y2_trad = x2*np.log(x2/x1) + x2/x1 * y1 # checked
    y2_forw = y1*x2/x1 + 2*x2 - 2*x1*np.sqrt(x2/x1) # checked
    y2_gt = x2**2/np.square(x1)*y1 + 2*x2**2/x1 - 2 *x2 # checked

    ax2 = plt.subplot(222)
    ax2.set_yscale('symlog')
    ax2.set_title(r'$f(x)=x, d(x)=-\frac{1}{2}x$', fontsize=24)
    ax2.plot(x1, y2_trad, label='trad', linewidth=2.5)
    ax2.plot(x1, y2_gt, label='fs', linewidth=2.5)
    ax2.plot(x1, y2_slb_e, label='slb', linewidth=2.5)
    ax2.plot(x1, y2_forw, label='fwd', linewidth=2.5)
    ax2.set_xlabel(r'$x_{1}$', fontsize=24)
    ax2.set_ylabel(r'$y_{2}$', fontsize=24)
    ax2.legend(fontsize=20)

    # DS:
    # x dot = x, y dot = y+x
    # x dot = x - 3/2x, y dot = y + x
    # only for positive systems.
    ax3 = plt.subplot(223)
    ax3.set_yscale('symlog')
    ax3.set_yticks([-1e3, -100, -10, 0, 1e1, 1e3, 1e5, 1e7, 1e9])
    y2_trad = x2*np.log(x2/x1) + x2/x1 * y1
    y2_gt = -2/3*x2 + 2/3 * x1**3/x2**2 + x1**2/x2**2 * y1
    y2_slb = x2 * np.log(x1**2/x2**2) + x1**2/x2**2 * y1
    # y2_forw = x2**(-1/2) / x1**(-3/2) * (-2/3) + 2/3*x2 + x2/x1*y1
    y2_forw = x2/x1 * (-2/3 * x1 * (x2/x1)**(-3/2) + 2/3*x1 + y1)

    ax3.set_title(r'$f(x)=x, d(x)=-\frac{3}{2}x$', fontsize=24)
    ax3.plot(x1, y2_trad, label='trad', linewidth=2.5)
    ax3.plot(x1, y2_gt, label='fs', linewidth=2.5)
    ax3.plot(x1, y2_slb, label='slb', linewidth=2.5)
    ax3.plot(x1, y2_forw, label='fwd', linewidth=2.5)
    ax3.set_xlabel(r'$x_{1}$', fontsize=24)
    ax3.set_ylabel(r'$y_{2}$', fontsize=24)
    ax3.legend(fontsize=20)

    # DS:
    # x dot = -x, y dot = y+x
    # x dot = -x + 2x, y dot = y+x
    # only for positive systems.
    y2_trad = (y1 + x1/2) * x1/x2 - x2/2
    y2_slb = -1/2 * x2 + 1/2 * x2**3/x1**2 + x2/x1 * y1
    y2_gt = x2 * np.log(x2/x1) + y1 * x2 / x1
    y2_forw = x1/x2 * (x1 * np.log(x1/x2) + y1)


    ax4 = plt.subplot(224)
    ax4.set_yscale('symlog')
    ax4.set_yticks([-100, -10, 0, 1e1, 1e3, 1e5, 1e7, 1e9])
    ax4.set_title(r'$f(x)=-x, d(x)=2x$', fontsize=24)
    ax4.plot(x1, y2_trad, label='trad', linewidth=2.5)
    ax4.plot(x1, y2_gt, label='fs', linewidth=2.5)
    ax4.plot(x1, y2_slb, label='slb', linewidth=2.5)
    ax4.plot(x1, y2_forw, label='fwd', linewidth=2.5)
    ax4.set_xlabel(r'$x_{1}$', fontsize=24)
    ax4.set_ylabel(r'$y_{2}$', fontsize=24)
    ax4.legend(fontsize=20)

    # plt.savefig('dx_fs_4.eps', dpi=300, bbox_inches='tight',pad_inches = 0.05)
    plt.show()
