# 3D グラフを描画する方法
# http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    print(X.shape)
    Z = X ** 2 + Y ** 2
    print(Z.shape)

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.plot_wireframe(X, Y, Z)
    fig.add_axes(ax)

    plt.show()
