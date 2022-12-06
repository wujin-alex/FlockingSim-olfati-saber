import numpy as np
import matplotlib.pyplot as plt


dt = 0.05            #更新周期，s
duration = 20      #测试时长，s
pos_goal = 5
num = int(duration / dt)

class Param:
    def __init__(self, p, i, d):
        self.p = p
        self.i = i
        self.d = d
    def __str__(self):
        return "p={},i={},d={}".format(self.p, self.i,self.d)

def pid(param):
    pos = 0
    vel = 0
    acc = 0
    dp_last = 0
    accumulate = 0
    pos_list = []
    for i in range(num):
        dp = pos_goal - pos
        if i == 0:
            acc = param.p * dp
        else:
            acc = param.p*dp + param.i*dp*dt + param.d*(dp-dp_last)/dt
        dp_last = dp
        vel += acc * dt
        pos += vel * dt

        pos_list.append(pos)
    return pos_list


def draw(axes, pos_list, label='pos'):
    goal = np.array([pos_goal]*num)
    pos = np.array(pos_list)

    axes.plot(pos, label=label)
    axes.plot(goal)
    axes.legend()

if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 10))
    num_fig = 1
    ##
    ax_pid1 = fig.add_subplot(num_fig,1,1)
    param1 = Param(p=0.2, i=0, d=0.6)
    pos1 = pid(param1)
    draw(ax_pid1, pos1, param1)

    ##
    ax_pid2 = fig.add_subplot(num_fig,1,1)
    param2 = Param(p=0.5, i=0, d=0.8)
    pos2 = pid(param2)
    # ax_pid2.set_title(param2)
    draw(ax_pid2, pos2, param2)

    ##
    ax_pid3 = fig.add_subplot(num_fig,1,1)
    param3 = Param(p=0.8, i=0, d=1.2)
    pos3 = pid(param3)
    # ax_pid3.set_title(param3)
    draw(ax_pid3, pos3, param3)

    plt.show()