import numpy as np
import matplotlib.pyplot as plt
import functions as ft
from params import *

def draw_sigma_norm(axes, param):
    num = 100                           # 绘图的点个数
    z = np.linspace(-10, 10, num)       # z，自变量，横轴坐标

    fun_sn = ft.sigma_norm(z, param.epsilon)
    fun_sng = ft.sigma_norm_gradient(z, param.epsilon)

    # 绘制sigma norm function曲线
    # axes.plot(z, fun_sn, label='sigma norm function')
    axes.plot(z, fun_sng, label='gradient')

    # 绘制y=0
    y0 = np.zeros(num)
    axes.plot(z, y0, label="y=0")
    # 画出图，添加图注释
    axes.set_title(label="sigma norm&gradient, epsilon={}".format(param.epsilon), loc='center')
    axes.set_title(label="gradient, epsilon={}".format(param.epsilon), loc='center')
    axes.legend()


def draw_bump_function(axes, param):
    num = 100                        # 绘图的点个数
    z = np.linspace(0, 2, num)       # z，自变量，横轴坐标

    fun_bump = ft.bump_function(z, param.h)
    # print(fun_bump.shape)
    
    # 绘制bump function曲线
    axes.plot(z, fun_bump, label='bump function')

    # 参数h
    yh = np.linspace(fun_bump[0],fun_bump[-1],10)
    xh = np.array([param.h]*10)
    axes.plot(xh, yh, label='h={:.2f}'.format(param.h))

    # 画出图，添加图注释
    axes.set_title(label="bump function, h={}".format(param.h), loc='center')
    axes.legend()

def draw_uneven_sigmoidal_fucntion(axes, param):
    num = 100
    z = np.linspace(-5, 5, num)

    fun_usf = ft.uneven_sigmoidal_fucntion(z, param.a, param.b)
    # fun_usf = list(map(ft.uneven_sigmoidal_fucntion, z.data, [param.a]*num, [param.b]*num))

    axes.plot(z, fun_usf, label='uneven sigmoidal')

    axes.set_title(label="uneven sigmoidal function, a={},b={}".format(param.a, param.b), loc='center')
    axes.legend()

def draw_action_function(axes, param):
    num = 100
    z = np.linspace(0, 20, num)

    fun_a = ft.action_function(z, param.epsilon, param.r, param.h, param.d, param.a, param.b)

    # 绘制action function曲线
    axes.plot(z, fun_a, label='action')

    ## 参数r
    sigma_r = ft.sigma_norm(param.r, param.epsilon)
    xr = np.array([sigma_r]*10)
    yr = np.linspace(fun_a[0], fun_a[-1], 10)
    axes.plot(xr, yr, label='sigma_r={:.2f}'.format(sigma_r))
    ## 参数d
    sigma_d = ft.sigma_norm(param.d, param.epsilon)
    xd = np.array([sigma_d]*10)
    yd = np.linspace(fun_a[0], fun_a[-1], 10)
    axes.plot(xd, yd, label='sigma_d={:.2f}'.format(sigma_d))

    axes.set_title(label="action function, eps={},r={},d={},a={},b={}".format(param.epsilon,param.r,param.d,param.a, param.b), loc='center')
    axes.legend()


Params.print_param()

def draw(params):
    fig = plt.figure(figsize=(10, 10))
    ax_sn = fig.add_subplot(2,2,1)
    ax_bf = fig.add_subplot(2,2,2)
    ax_af = fig.add_subplot(2,2,3)
    ax_usf = fig.add_subplot(2,2,4)

    draw_sigma_norm(ax_sn, params)
    draw_bump_function(ax_bf, params)
    draw_action_function(ax_af, params)
    draw_uneven_sigmoidal_fucntion(ax_usf, params)



if __name__ == '__main__':
    print("draw")
    draw(Params)
    draw(Params2)
    plt.show()