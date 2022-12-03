import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

import functions as ft
import draw as dr
from utils import *
from params import *

Params.print_param()
flag_print = False                                # 是否进行调试信息打印

sim_dim = 2                                       # 仿真维度（根据需要设置），根据需要进行设置，可选择2D或者3D
num_agent = 10                                    # 仿真智能体数量（根据需要设置）
max_acc = 20        # 最大加速度m^2/s
max_vel = 10        # 最大速度m/s
flag_init = [False]*num_agent

interval = 50
dt = 1.0/interval
cnt = 0

flag_draw_arrow_acc = False                       # 画图控制（根据需要设置），是否绘制智能体加速度向量
flag_draw_arrow_vel = True                       # 画图控制（根据需要设置），是否绘制智能体速度向量
flag_draw_neighbor_line = False                    # 画图控制（根据需要设置），是否绘制智能体与邻居节点的连线
## 颜色
# c_list= np.random.rand(num_agent) * 255
c_list = np.linspace(0, 255, num_agent)           # 颜色列表，用于绘制

if sim_dim == 2:
    area_range = (10,10)                          # 区域大小，10x10m
    pos_desire = np.array([10,10])                # 集群导航，集群期望位置，可动态变动
    vel_desire = np.array([0,0])                  # 集群导航，集群期望速度，期望位置的导数
elif sim_dim == 3:
    area_range = (10,10,10)                       # 区域大小，10x10x10m
    pos_desire = np.array([-1,1,2])               # 集群导航，集群期望位置，可动态变动
    vel_desire = np.array([0,0,0])                # 集群导航，集群期望速度，期望位置的导数

##
agent_pos = np.zeros((sim_dim, num_agent))          # 位置矩阵，dimxn
agent_vel = np.zeros((sim_dim, num_agent))          # 速度矩阵，dimxn
agent_acc = np.zeros((sim_dim, num_agent))          # 加速度矩阵，dimxn
agent_dpos_last = np.zeros((sim_dim, num_agent))

##
N_idx    = []                                       # 保存每个智能体的邻居id
N_01     = np.zeros((num_agent,num_agent))          # 保存连接关系，Nij表示智能体i与j的连接关系，0代表不连接，1代表连接
N_bump   = np.zeros((num_agent,num_agent))          # 保存邻接矩阵，参考公式(11)，bump(sigma(||qj-qi||)/sigma(r))所有元素都在0~1
N_sigma  = np.zeros((num_agent,num_agent))          # 根据智能体i与j的距离，经过sigma(||qj-qi||)处理
N_action = np.zeros((num_agent,num_agent))          # 根据智能体i与j的距离，计算action

def algo(agent_pos, agent_vel, agent_acc):
    """
    算法olfati saber
    """
    global cnt, flag_print, flag_init, N_01, N_bump, N_sigma, N_action

    my_print(flag_print, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> cnt={}".format(cnt))
    cnt += 1

    ## 重置
    N_idx    = []
    N_01     = np.zeros((num_agent,num_agent))
    N_bump   = np.zeros((num_agent,num_agent))
    N_sigma  = np.zeros((num_agent,num_agent))
    N_action = np.zeros((num_agent,num_agent))

    for i in range(num_agent):
        N_i = []
        
        u_sum = np.zeros(sim_dim)
        u_g_sum = np.zeros(sim_dim)
        u_v_sum = np.zeros(sim_dim)

        qi = agent_pos[:,i]
        pi = agent_vel[:,i]
        my_print(flag_print, "qi={}, pi={}".format(qi, pi))
        for j in range(num_agent):
            my_print(flag_print, "--- i={}, j={}".format(i, j))
            if i==j:
                continue
            qj = agent_pos[:,j]
            pj = agent_vel[:,j]
            dq = qj - qi              # 计算agent-i与agent-j的距离
            dis = np.linalg.norm(dq)  # 计算agent-i与agent-j的距离绝对值
            # my_print(flag_print, "qj={}, pj={}".format(qj, pj))
            # my_print(cur_print_flag, "dq={}, dis={}".format(dq, dis))

            if dis <= Params.r:
                sigma_dq = ft.sigma_norm(qj-qi, Params.epsilon)
                ## 计算梯度项
                u_g_af = ft.action_function(ft.sigma_norm(qj-qi, Params.epsilon), Params.epsilon, Params.r, Params.h, Params.d, Params.a, Params.b)
                u_g_nij = ft.nij(qi, qj, Params.epsilon)
                u_g = u_g_af * u_g_nij
                my_print(flag_print, "u_g={}*{}={}".format(u_g_af, u_g_nij, u_g))
                u_g_sum += u_g

                ## 计算速度项
                u_v_a = ft.aij(qi, qj, Params.epsilon, Params.r, Params.h)
                u_v_p = (pj - pi)
                u_v = u_v_a * u_v_p
                my_print(flag_print, "u_v={}*{}={}".format(u_v_a, u_v_p, u_v))
                u_v_sum += u_v

                ## 若距离小于r，说明j是i的邻居
                N_i.append(j)
                N_01[i,j] = 1    
                N_bump[i,j] = ft.bump_function(sigma_dq/ft.sigma_norm(Params.r, Params.epsilon), Params.h)
                N_sigma[i,j] = sigma_dq
                N_action[i,j] = u_g_af

        my_print(flag_print, " ")
        ## 更新邻接矩阵
        N_idx.append(N_i)

        ## 计算位置项
        dpos = pos_desire - qi
        if flag_init[i]:
            dpos_last = agent_dpos_last[:,i]
            u_p = Params.pid_p * dpos + Params.pid_d * (dpos - dpos_last) / dt
        else:
            flag_init[i] = True
            u_p = Params.pid_p * dpos
        agent_dpos_last[:,i] = dpos

        ## 计算加速度
        u_sum = u_g_sum + u_v_sum + u_p

        ## 更新智能体运动状态
        agent_acc[:,i] = u_sum
        if np.linalg.norm(agent_acc[:,i]) > max_acc:
            agent_acc[:,i] = agent_acc[:,i] / np.linalg.norm(agent_acc[:,i]) * max_acc

        agent_vel[:,i] += u_sum * dt
        if np.linalg.norm(agent_vel[:,i]) > max_acc:
            agent_vel[:,i] = agent_vel[:,i] / np.linalg.norm(agent_vel[:,i]) * max_vel
        
        agent_pos[:,i] += agent_vel[:,i] * dt

def draw_action(axes, param, agent_id):
    """
    根据一个智能体与邻居的距离，绘制对应action function曲线上的位置
    axes: 绘图板
    param: 参数
    agent_id: 智能体ID
    """
    num = 100
    x = np.linspace(0, 20, num)
    ## 绘制action曲线
    y_a = ft.action_function(x, param.epsilon, param.r, param.h, param.d, param.a, param.b)
    axes.plot(x, y_a, label='action')
    ## 绘制r,d所在的位置
    sigma_d = ft.sigma_norm(param.d, param.epsilon)
    sigma_d_y = ft.action_function(sigma_d, param.epsilon, param.r, param.h, param.d, param.a, param.b)
    sigma_r = ft.sigma_norm(param.r, param.epsilon)
    sigma_r_y = ft.action_function(sigma_r, param.epsilon, param.r, param.h, param.d, param.a, param.b)
    axes.scatter(sigma_d, sigma_d_y, s=80, c='red', marker='*', label="d={:.2f}, sd={:.2f}".format(param.d, sigma_d))
    axes.scatter(sigma_r, sigma_r_y, s=80, c='red', marker='o', label="r={:.2f}, sr={:.2f}".format(param.r, sigma_r))

    ## 根据智能体各邻居节点的距离，绘制在action曲线上的位置
    mask = N_01[agent_id,:] > 0
    x_, y_ = N_sigma[agent_id, mask], N_action[agent_id, mask]
    axes.scatter(x_, y_, c=c_list[mask], marker='v')

    idx = np.linspace(0, num_agent-1, num_agent)[mask].astype(int)
    for id,tx,ty in zip(idx, x_, y_):
        axes.text(tx, ty, '{}'.format(id))

    ## 显示该智能体的速度和加速度大小
    vel = np.linalg.norm(agent_vel[:,agent_id])
    acc = np.linalg.norm(agent_acc[:,agent_id])
    axes.text(2, -2, "acc={:.2f}, vel={:.2f}".format(acc, vel))


def sim_loop():
    np.random.seed(19680801)
    if sim_dim == 2:
        lim_scale = 2
        # agent_pos = np.dot(np.diag(np.array((1,1))), np.random.rand(sim_dim, num_agent))
    elif sim_dim == 3:
        lim_scale = 1
        # agent_pos = np.dot(np.diag(np.array((1,1,1))), np.random.rand(sim_dim, num_agent))     # 将智能体初始位置靠在1x1m的区域内

    agent_pos = np.dot(np.diag(np.array(area_range)), np.random.rand(sim_dim, num_agent))

    ### 绘图
    fig = plt.figure(figsize=(8,10))
    if sim_dim == 2:
        ax = fig.add_subplot(2,1,1)
        ax_action = fig.add_subplot(2,1,2)
    elif sim_dim == 3:
        gs = gridspec.GridSpec(3, 1)
        ax = fig.add_subplot(gs[:2,0], projection='3d')
        # ax = Axes3D(fig)  
        ax_action = fig.add_subplot(gs[2,0])
    ax.set_xlim(-area_range[0]*lim_scale, area_range[0]*lim_scale)
    ax.set_ylim(-area_range[1]*lim_scale, area_range[1]*lim_scale)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if sim_dim == 3:
        ax.set_zlabel('Z')
        ax.set_zlim(-area_range[2]*lim_scale, area_range[2]*lim_scale)

    def update(n):
        ## 运行集群算法，得到更新后的位置
        algo(agent_pos, agent_vel, agent_acc)

        ## 绘图
        # 清空
        ax.clear()
        ax_action.clear()
        # 设置范围
        ax.set_xlim(-area_range[0]*lim_scale, area_range[0]*lim_scale)
        ax.set_ylim(-area_range[1]*lim_scale, area_range[1]*lim_scale)
        if sim_dim == 3:
            ax.set_zlim(-area_range[2]*lim_scale, area_range[2]*lim_scale)

        # 绘制智能体位置和连线
        if sim_dim == 2:
            # 绘制集群导航期望位置
            ax.scatter(pos_desire[0], pos_desire[1], s=80, c='red', marker='p')
            # 绘制智能体位置
            scatter1 = ax.scatter(agent_pos[0,:], agent_pos[1,:], c=c_list)
            for idx, (tx,ty) in enumerate(zip(agent_pos[0,:], agent_pos[1,:])):
                ax.text(tx, ty, '{}'.format(idx))
            # 绘制智能体与邻居的连线
            if flag_draw_neighbor_line:
                for i in range(num_agent):
                    for j in range(i, num_agent):
                        if N_01[i,j]>0:
                            x,y = zip(agent_pos[:,i], agent_pos[:,j])
                            ax.plot(x,y)
        elif sim_dim == 3:
            # 绘制集群导航期望位置
            ax.scatter(pos_desire[0], pos_desire[1], pos_desire[2], s=80, c='red', marker='p')
            # 绘制智能体位置
            scatter1 = ax.scatter3D(agent_pos[0,:], agent_pos[1,:], agent_pos[2,:], c=c_list)
            for idx, (tx,ty,tz) in enumerate(zip(agent_pos[0,:], agent_pos[1,:], agent_pos[2,:])):
                ax.text(tx, ty, tz, '{}'.format(idx))
            # 绘制智能体与邻居的连线
            if flag_draw_neighbor_line:
                for i in range(num_agent):
                    for j in range(i, num_agent):
                        if N_01[i,j]>0:
                            x,y,z = zip(agent_pos[:,i], agent_pos[:,j])
                            ax.plot(x,y,z)

        # 绘制箭头
        X = agent_pos[0,:]
        Y = agent_pos[1,:]
        U_v = agent_vel[0,:]
        V_v = agent_vel[1,:]
        U_a = agent_acc[0,:]
        V_a = agent_acc[1,:]
        if sim_dim == 2:
            if flag_draw_arrow_vel:
                ax.quiver(X, Y, U_v, V_v, scale=20, color="blue")        # 绘制速度箭头 scale是缩小比例
            if flag_draw_arrow_acc:
                ax.quiver(X, Y, U_a, V_a, scale=30, color="green")       # 绘制加速度箭头
        elif sim_dim == 3:
            Z = agent_pos[2,:]
            W_v = agent_vel[2,:]
            W_a = agent_acc[2,:]
            if flag_draw_arrow_vel:
                ax.quiver3D(X, Y, Z, U_v, V_v, W_v, normalize=False, color="blue")         # 绘制速度箭头 scale是缩小比例
            if flag_draw_arrow_acc:
                ax.quiver3D(X, Y, Z, U_a, V_a, W_a, normalize=False, color="green")        # 绘制加速度箭头

        # 绘制action曲线
        draw_action(ax_action, Params, 8)                                # 选择一个智能体绘制action曲线
        
        ax.legend()
        ax_action.legend()

    ani = FuncAnimation(fig, update, frames=500, interval=interval, blit=False, repeat=False)     # interval两帧间隔时间单位是ms
    
    plt.show()


if __name__ == '__main__':
    print("simulation olfati saber flocking algo")
    sim_loop()