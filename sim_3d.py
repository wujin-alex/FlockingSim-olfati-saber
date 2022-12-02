import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import functions as ft
import draw as dr

## parameters
class Params:
    epsilon = 0.1            # epsilon<<1
    h = 0.3                  # h in (0,1)
    c1 = 0.3                 # c1>0
    c2 = 0.2                 # c2>0
    a = 1
    b = 2
    # 参数r和d
    r = 6                  # 有效距离
    d = 4                  # 智能体间距

    def print_param():
        print("sigma_d={}, sigma_r={}".format(ft.sigma_norm(Params.d, Params.epsilon), ft.sigma_norm(Params.r, Params.epsilon)))

    def __str__(self):
        return "epsilon={},h={},c1={},c2={},a={},b={},r={},d={}".format(self.epsilon,self.h,
                                                                        self.c1,self.c2,
                                                                        self.a,self.b,
                                                                        self.r,self.d)
Params.print_param()


num_agent = 3
area_range = (10,10,5)                  # 区域大小，10x10x5m
agent_pos = np.zeros((3, num_agent))
agent_vel = np.zeros((3, num_agent))
agent_acc = np.zeros((3, num_agent))

interval = 100
dt = 1.0/interval
cnt = 0

def algo(agent_pos, agent_vel, agent_acc):
    global cnt

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> cnt={}".format(cnt))
    cnt += 1

    for i in range(num_agent):
        N_i = []
        
        u_sum = np.zeros(3)
        qi = agent_pos[:,i]
        pi = agent_vel[:,i]
        print("qi={}, pi={}".format(qi, pi))
        for j in range(num_agent):
            print("--- i={}, j={}".format(i, j))
            if i==j:
                continue
            qj = agent_pos[:,j]
            pj = agent_vel[:,j]
            dis = np.linalg.norm(qi - qj)   # 计算agent-i与agent-j的距离
            print("qj={}, pj={}".format(qj, pj))
            print("dq={}, dis={}".format(qj-qi, dis))

            if dis <= Params.r:
                N_i.append(j)
                ## 计算梯度项
                u_g_af = ft.action_function(ft.sigma_norm(qj-qi, Params.epsilon), Params.epsilon, Params.r, Params.h, Params.d, Params.a, Params.b)
                u_g_nij = ft.nij(qi, qj, Params.epsilon)
                u_g = u_g_af * u_g_nij
                print("u_g={}*{}={}".format(u_g_af, u_g_nij, u_g))
                ## 计算速度项
                u_v_a = ft.aij(qi, qj, Params.epsilon, Params.r, Params.h)
                u_v_p = (pj - pi)
                u_v = u_v_a * u_v_p
                print("u_v={}*{}={}".format(u_v_a, u_v_p, u_v))
                u_sum += u_g + u_v
        print(" ")

        ## 计算位置项
        u_p = np.zeros(3)

        ## 计算加速度
        u_sum += u_p

        ## 更新速度和位置
        agent_acc[:,i] = u_sum
        agent_vel[:,i] += u_sum * dt
        agent_pos[:,i] += agent_vel[:,i] * dt

def sim_loop():
    agent_pos = np.dot(np.diag(np.array(area_range)), np.random.rand(3, num_agent))

    np.random.seed(19680801)

    ### 绘图
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
    ax.legend()
    ax.set_xlim(0, area_range[0]*5)
    ax.set_ylim(0, area_range[1]*5)
    ax.set_zlim(0, area_range[2]*5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ## 颜色
    # c_list= np.random.rand(num_agent) * 255
    c_list = np.linspace(0, 255, num_agent)

    def update(n):
        ## 运行集群算法，得到更新后的位置
        algo(agent_pos, agent_vel, agent_acc)

        ## 绘图
        ax.clear()
        ax.set_xlim(0, area_range[0]*5)
        ax.set_ylim(0, area_range[1]*5)
        ax.set_zlim(0, area_range[2]*5)
        scatter1 = ax.scatter3D(agent_pos[0,:], agent_pos[1,:], agent_pos[2,:], c=c_list)

    ani = FuncAnimation(fig, update, frames=100, interval=100, blit=False, repeat=False)     # interval两帧间隔时间单位是ms
    
    plt.show()


def sim_loop_bak():
    num_agent = 10
    area_range = (10,10,5)                  # 区域大小，10x10x5m
    agent_pos = np.zeros((3, num_agent))
    agent_vel = np.zeros((3, num_agent))
    agent_acc = np.zeros((3, num_agent))

    agent_pos = np.dot(np.diag(np.array(area_range)), np.random.rand(3, num_agent))

    np.random.seed(19680801)

    ### 绘图
    # ax = plt.figure().add_subplot(projection='3d')
    ax = plt.axes(projection='3d')
    ax.legend()
    ax.set_xlim(0, area_range[0])
    ax.set_ylim(0, area_range[1])
    ax.set_zlim(0, area_range[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ## 颜色
    # c_list= np.random.rand(num_agent) * 255
    c_list = np.linspace(0, 255, num_agent)

    plt.ion()
    for i in range(100):

        algo(agent_pos, agent_vel, agent_acc)

        ax.scatter3D(agent_pos[0,:], agent_pos[1,:], agent_pos[2,:], c=c_list)
        plt.pause(0.1)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    print("simulation olfati saber flocking algo")
    # sim_loop()