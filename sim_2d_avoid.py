"""
二维平面内集群仿真。
1、实现集群智能体之间的避碰与速度协同；
2、实现避障障碍物；
3、实现飞行至期望位置（导航位置）。
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import functions as ft
import draw as dr
from utils import *
from params import *

## 设置随机seed
np.random.seed(19680801)

params = Params_obs
params.print_param()
flag_print = False

sim_dim = 2                                   # 仿真维度，2维
num_agent = 10                                # 仿真智能体数量
area_range = (20,20)                          # 区域大小，10x10m

pos_desire = np.array([25,10])                # 集群导航，集群期望位置，可动态变动
vel_desire = np.array([0,0])                  # 集群导航，集群期望速度，期望位置的导数

max_acc = 20        # 最大加速度m^2/s
max_vel = 10        # 最大速度m/s

flag_init = [False]*num_agent

interval = 50
dt = 1.0/interval
cnt = 0

flag_draw_arrow_acc = False                       # 画图控制（根据需要设置），是否绘制智能体加速度向量
flag_draw_arrow_vel = False                       # 画图控制（根据需要设置），是否绘制智能体速度向量
flag_draw_neighbor_line = True                   # 画图控制（根据需要设置），是否绘制智能体与邻居节点的连线
flag_draw_obstacle_line = True                   # 画图控制（根据需要设置），是否绘制智能体与障碍物的连线
## 颜色
# c_list= np.random.rand(num_agent) * 255
c_list = np.linspace(0, 255, num_agent)

##
agent_pos = np.zeros((sim_dim, num_agent))    # 位置矩阵，2xn
agent_vel = np.zeros((sim_dim, num_agent))    # 速度矩阵，2xn
agent_acc = np.zeros((sim_dim, num_agent))    # 加速度矩阵，2xn
agent_dpos_last = np.zeros((sim_dim, num_agent))

##
N_idx    = []                                       # 保存每个智能体的邻居id
N_01     = np.zeros((num_agent,num_agent))          # 保存连接关系，Nij表示智能体i与j的连接关系，0代表不连接，1代表连接
N_bump   = np.zeros((num_agent,num_agent))          # 保存邻接矩阵，参考公式(11)，bump(sigma(||qj-qi||)/sigma(r))所有元素都在0~1
N_sigma  = np.zeros((num_agent,num_agent))          # 根据智能体i与j的距离，经过sigma(||qj-qi||)处理
N_action = np.zeros((num_agent,num_agent))          # 根据智能体i与j的距离，计算action


## 设置障碍物
num_obstacles = 2
# obs_sphere_center = np.dot(np.diag(np.array((10,10))), np.random.randn(sim_dim, num_obstacles))
# obs_sphere_radius = np.random.randint(1,5, size=num_obstacles)
obs_sphere_center = np.array([[10, 10],
                            [-5, 20]])
obs_sphere_radius = np.array([5, 4])

N_beta_01  = np.zeros((num_agent, num_obstacles))          # 保存智能体与障碍物连接关系，Nik表示智能体i与障碍物k的连接关系，0代表不连接，1代表连接
N_beta_pos = np.zeros((num_agent, num_obstacles, sim_dim))   # b-agent位置，第i行表示所有障碍物相对于第i个智能体的b-agent（这里暂不考虑距离限制条件）
N_beta_vel = np.zeros((num_agent, num_obstacles, sim_dim))   # b-agent速度

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

    cur_print_flag = False
    for i in range(num_agent):
        if(i in (8,)):
            cur_print_flag = True
            my_print(True, "--------------------------------------- agent_id={}".format(i))
        else:
            cur_print_flag = False
        N_i = []
        
        u_sum = np.zeros(sim_dim)

        qi = agent_pos[:,i]
        pi = agent_vel[:,i]
        my_print(flag_print, "qi={}, pi={}".format(qi, pi))

        ''' 1、协同与避碰，alpha-alpha agent '''
        u_alpha = np.zeros(sim_dim)
        u_g_sum = np.zeros(sim_dim)
        u_v_sum = np.zeros(sim_dim)
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

            if dis <= params.r:
                sigma_dq = ft.sigma_norm(qj-qi, params.epsilon)
                ## 计算梯度项
                u_g_af = ft.action_function(sigma_dq, params.epsilon, params.r, params.h, params.d, params.a, params.b)
                u_g_nij = ft.nij(qi, qj, params.epsilon)
                u_g = u_g_af * u_g_nij
                my_print(flag_print, "u_g={}*{}={}".format(u_g_af, u_g_nij, u_g))
                u_g_sum += params.c1_alpha * u_g

                ## 计算速度项
                u_v_a = ft.aij(qi, qj, params.epsilon, params.r, params.h)
                u_v_p = (pj - pi)
                u_v = u_v_a * u_v_p
                my_print(flag_print, "u_v={}*{}={}".format(u_v_a, u_v_p, u_v))
                u_v_sum += params.c2_alpha * u_v

                ## 若距离小于r，说明j是i的邻居
                N_i.append(j)
                N_01[i,j] = 1        
                N_bump[i,j] = ft.bump_function(sigma_dq/ft.sigma_norm(params.r, params.epsilon), params.h)
                N_sigma[i,j] = sigma_dq
                N_action[i,j] = u_g_af
                
                # my_print(cur_print_flag, "u_g={:.2f}, u_v={:.2f}".format(np.linalg.norm(u_g), np.linalg.norm(u_v)))
                # my_print(cur_print_flag, "u_g_sum={}, u_v_sum={}".format(u_g_sum, u_v_sum))
        u_alpha = u_g_sum + u_v_sum

        # my_print(i==8, "---------N_01")
        # my_print(i==8, N_01)
        # my_print(i==8, "---------N_bump")
        # my_print(i==8, N_bump)
        # my_print(i==8, "---------N_action")
        # my_print(i==8, N_action)

        my_print(flag_print, " ")
        ## 更新邻接矩阵
        N_idx.append(N_i)

        '''2、避障，alpha-beta agent '''
        u_beta = np.zeros(sim_dim)
        # 计算障碍物位置
        for k in range(num_obstacles):
            qik, pik = ft.get_beta_agent_sphere(agent_pos[:,i], agent_vel[:,i], obs_sphere_center[:,k], obs_sphere_radius[k])
            # print(">>>>>>>>>>>", qik, pik)
            N_beta_pos[i,k,:] = qik
            N_beta_vel[i,k,:] = pik
            dq_obs = qi - qik
            dis = np.linalg.norm(dq_obs)
            if dis < params.r_obs:
                N_beta_01[i,k] = 1
                sigma_dq = ft.sigma_norm(qik-qi, params.epsilon)
                ## 计算梯度项（障碍物）
                u_beta_g = ft.repulsive_action_function(ft.sigma_norm(qik-qi, params.epsilon), params.epsilon, params.d_obs, params.h) * ft.nij(qi, qik, params.epsilon)
                ## 计算速度项
                u_beta_v = ft.bik(qi, qik, params.epsilon, params.h, params.d_obs) * (pik - pi)
                u_beta += params.c1_beta * u_beta_g + params.c2_beta * u_beta_v

        ''' 3、导航：alpha-gamma agent '''
        u_gamma = np.zeros(sim_dim)
        # dpos = pos_desire - qi
        # if flag_init[i]:
        #     dpos_last = agent_dpos_last[:,i]
        #     u_gamma = params.pid_p * dpos + params.pid_d * (dpos - dpos_last) / dt
        # else:
        #     flag_init[i] = True
        #     u_gamma = params.pid_p * dpos
        # my_print(cur_print_flag, "dpos    ={:.2f}, dpos=[{:.2f} {:.2f}]".format(np.linalg.norm(dpos), dpos[0], dpos[1]))
        # my_print(cur_print_flag, "u_gamma_norm={:.2f}, u_gamma =[{:.2f} {:.2f}]".format(np.linalg.norm(u_gamma), u_gamma[0], u_gamma[1]))
        # agent_dpos_last[:,i] = dpos

        u_gamma = -params.c1 * ft.sigma_norm_gradient(qi-pos_desire, epsilon=1) - params.c2 * (pi - vel_desire)

        ## 计算加速度
        u_sum = u_alpha + u_beta + u_gamma

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

    ## 
    mask = N_01[agent_id,:] > 0
    # print("-----------------------------------", N_action[agent_id, mask])
    x_, y_ = N_sigma[agent_id, mask], N_action[agent_id, mask]
    axes.scatter(x_, y_, c=c_list[mask], marker='v')

    idx = np.linspace(0, num_agent-1, num_agent)[mask].astype(int)
    for id,tx,ty in zip(idx, x_, y_):
        axes.text(tx, ty, '{}'.format(id))

    ##
    vel = np.linalg.norm(agent_vel[:,agent_id])
    acc = np.linalg.norm(agent_acc[:,agent_id])
    axes.text(2, -2, "acc={:.2f}, vel={:.2f}".format(acc, vel))


def sim_loop():
    lim_scale = 2
    # agent_pos = np.dot(np.diag(np.array(area_range)), np.random.randn(sim_dim, num_agent))
    agent_pos = np.dot(np.diag(np.array((-15, 10))), np.random.randn(sim_dim, num_agent))


    ### 绘图
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(2,1,1)
    # ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(-area_range[0]*lim_scale, area_range[0]*lim_scale)
    ax.set_ylim(-area_range[1]*lim_scale, area_range[1]*lim_scale)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax_action = fig.add_subplot(2,1,2)

    def update(n):
        ## 运行集群算法，得到更新后的位置
        algo(agent_pos, agent_vel, agent_acc)

        ## 绘图
        ax.clear()
        ax_action.clear()

        ax.set_xlim(-area_range[0]*lim_scale, area_range[0]*lim_scale)
        ax.set_ylim(-area_range[1]*lim_scale, area_range[1]*lim_scale)
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
        
        # 绘制箭头
        X = agent_pos[0,:]
        Y = agent_pos[1,:]
        U_v = agent_vel[0,:]
        V_v = agent_vel[1,:]
        U_a = agent_acc[0,:]
        V_a = agent_acc[1,:]
        if flag_draw_arrow_vel:
            ax.quiver(X, Y, U_v, V_v, scale=10, color="blue")         # 绘制速度箭头 scale是缩小比例
        if flag_draw_arrow_acc:
            ax.quiver(X, Y, U_a, V_a, scale=20, color="yellow")       # 绘制加速度箭头

        # 绘制action曲线
        draw_action(ax_action, params, 0)                         # 选择一个智能体绘制
        
        ## 绘制障碍物
        for k in range(num_obstacles):
            x = obs_sphere_center[0,k]
            y = obs_sphere_center[1,k]
            draw_circle = plt.Circle((x, y), 1, fill=True)
            draw_circle = plt.Circle((x, y), obs_sphere_radius[k], fill=True)
            ax.add_artist(draw_circle)
        # 在每个障碍物上绘制与智能体对应的beta-agent
        agent_beta_pos = N_beta_pos.reshape(-1, sim_dim).T
        scatter2 = ax.scatter(agent_beta_pos[0,:], agent_beta_pos[1,:], c=np.repeat(c_list, num_obstacles), marker='*')

        # 连接智能体与障碍物
        for i in range(num_agent):
            for k in range(num_obstacles):
                if N_beta_01[i,k] > 0:
                    x,y = zip(agent_pos[:,i], N_beta_pos[i,k])
                    ax.plot(x, y)

        ## legend
        ax.legend()
        ax_action.legend()

    ani = FuncAnimation(fig, update, frames=500, interval=interval, blit=False, repeat=False)     # interval两帧间隔时间单位是ms
    
    plt.show()

if __name__ == '__main__':
    print("simulation olfati saber flocking algo")
    sim_loop()