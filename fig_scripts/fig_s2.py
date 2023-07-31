import matplotlib
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import lambertw

mpl.rcParams['agg.path.chunksize'] = 10000


# DS:
# no disturbance: x dot = x, y dot = y+x
# with disturbance: x dot = x+a*t, y dot = y+x
def y2_trad(x1, x2, y1):
    t = np.log(x2/x1)
    y2 = y1 * np.exp(t) + t * np.exp(t) * x1

    return y2

def ttrue(x1, x2):
    pl = lambertw(-(a+x1)*np.exp(-1-x2/a) / a, k=-1)
    true = (-a - x2 - a*pl) / a

    return true

def y2_gt(x1, x2, y1):
    t = ttrue(x1, x2)
    return a*t*np.exp(t) + a*t + 2*a + (y1-2*a)*np.exp(t) + np.exp(t)*t*x1

def y2_slb(x1, x2, y1):
    y2 = y2_gt(x1, x2, y1)
    tif = (y2-x2*lambertw(np.exp(y2/x2)*y1/x2, k=0)) / x2

    true = ttrue(x1, x2)
    dt = true-tif
    xslb = -a * dt - a + (x1+a)*np.exp(dt)

    return xslb

def y2_forw(x1, x2, y1):
    t = np.log(x2/x1)
    return a*t*np.exp(t) + a*t + 2*a + (y1-2*a)*np.exp(t) + np.exp(t)*t*x1


if __name__ == '__main__':

    b = 1
    d = 1
    a = 1
    x2 = 100 #100
    y1 = 3
    
    x1 = np.linspace(0.1, x2, 2000)


    # for dt
    matplotlib.rc('xtick', labelsize=26)
    matplotlib.rc('ytick', labelsize=26)
    fig = plt.figure(figsize=(21, 6), facecolor='w', edgecolor='k') # for dt
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98, wspace=0.25, hspace=0.40)


    a = 1
    ax1 = plt.subplot(121)
    ax1.set_title('$a=1$', fontsize=30)
    ax1.plot(x1, np.log10(y2_trad(x1,x2,y1)), label='trad',linewidth=2.5)
    ax1.plot(x1, np.log10(y2_gt(x1,x2,y1)), label='fs', linewidth=2.5)
    ax1.plot(y2_slb(x1, x2, y1), np.log10(y2_gt(x1,x2,y1)), label='slb', linewidth=2.5)
    ax1.plot(x1, np.log10(y2_forw(x1,x2,y1)), label='fwd', linewidth=2.5)

    ax1twin = ax1.twinx()
    x_slb = y2_slb(x1, x2, y1)
    ygt = np.log10(y2_gt(x_slb, x2, y1))
    ytrad = np.log10(y2_trad(x_slb, x2, y1))
    yfwd = np.log10(y2_forw(x_slb, x2, y1))
    yslb = np.log10(y2_gt(x1,x2,y1))
    ax1twin.plot(x_slb, ytrad-ygt, label='trad-fs',linewidth=2.5, color=(26/255, 40/255, 71/255))
    ax1twin.plot(x_slb, yslb-ygt, label='slb-fs',linewidth=2.5, color=(61/255, 142/255, 134/255))
    ax1twin.plot(x_slb, yfwd-ygt, label='fwd-fs',linewidth=2.5, color=(225/255, 210/255, 121/255))

    # print('slb x1: ', y2_slb(x1, x2, y1).shape)
    # ax1.set_yscale('log')
    ax1.set_xlabel('$x_{1}$', fontsize=26)
    ax1twin.set_ylabel('$log_{10}(y_{2})-log_{10}(y_{2fs})$', fontsize=26)
    ax1.set_ylabel('$log_{10}(y_{2})$', fontsize=26)
    ax1.legend(fontsize=26)
    ax1twin.legend(fontsize=26, loc='lower right')


    a = 10
    ax2 = plt.subplot(122)
    ax2.set_title('$a=10$', fontsize=30)
    ax2.plot(x1, np.log10(y2_trad(x1,x2,y1)), label='trad',linewidth=2.5)
    ax2.plot(x1, np.log10(y2_gt(x1,x2,y1)), label='fs', linewidth=2.5)
    ax2.plot(y2_slb(x1, x2, y1), np.log10(y2_gt(x1,x2,y1)), label='slb', linewidth=2.5)
    ax2.plot(x1, np.log10(y2_forw(x1,x2,y1)), label='fwd', linewidth=2.5)
    ax2.set_xlabel('$x_{1}$', fontsize=26)
    ax2.set_ylabel('$log_{10}(y_{2})$', fontsize=26)
    ax2.legend(fontsize=26)

    plt.show()
