import functions as ft

## parameters
class Params:
    epsilon = 0.1            # epsilon<<1
    h = 0.3                  # h in (0,1)
    c1 = 0.05                 # c1>0
    c2 = 0.05                 # c2>0
    a = 1
    b = 5
    # 参数r和d
    r = 6                  # 有效距离
    d = 4                  # 智能体间距

    pid_p = c1             # 位置控制环P参数
    pid_d = 0.5            # 位置控制环D参数

    def print_param():
        print("sigma_d={}, sigma_r={}".format(ft.sigma_norm(Params.d, Params.epsilon), ft.sigma_norm(Params.r, Params.epsilon)))

    def __str__(self):
        return "epsilon={},h={},c1={},c2={},a={},b={},r={},d={}".format(self.epsilon,self.h,
                                                                        self.c1,self.c2,
                                                                        self.a,self.b,
                                                                        self.r,self.d)
class Params2:
    epsilon = 0.5            # epsilon<<1
    h = 0.3                  # h in (0,1)
    c1 = 0.05                 # c1>0
    c2 = 0.05                 # c2>0
    a = 1
    b = 5
    # 参数r和d
    r = 6                  # 有效距离
    d = 4                  # 智能体间距

    pid_p = c1             # 位置控制环P参数
    pid_d = 0.5            # 位置控制环D参数
    
    def print_param():
        print("sigma_d={}, sigma_r={}".format(ft.sigma_norm(Params.d, Params.epsilon), ft.sigma_norm(Params.r, Params.epsilon)))

    def __str__(self):
        return "epsilon={},h={},c1={},c2={},a={},b={},r={},d={}".format(self.epsilon,self.h,
                                                                        self.c1,self.c2,
                                                                        self.a,self.b,
                                                                        self.r,self.d)


class Params_obs:
    """
    包含障碍物
    """
    epsilon = 0.1            # epsilon<<1
    h = 0.3                  # h in (0,1)
    c1 = 1                 # c1>0
    c2 = 0.1                 # c2>0
    a = 1
    b = 5
    # 参数r和d
    r = 6                  # 有效距离
    d = 4                  # 智能体间距
    #
    pid_p = c1             # 位置控制环P参数
    pid_d = c2             # 位置控制环D参数

    # alpha-alpha agent
    c1_alpha = 1
    c2_alpha = 1

    # alpha-beta agent(与障碍物相关)
    r_obs = 8
    d_obs = 8
    c1_beta = 1
    c2_beta = 1


    def print_param():
        print("sigma_d={}, sigma_r={}".format(ft.sigma_norm(Params.d, Params.epsilon), ft.sigma_norm(Params.r, Params.epsilon)))

    def __str__(self):
        return "epsilon={},h={},c1={},c2={},a={},b={},r={},d={}".format(self.epsilon,self.h,
                                                                        self.c1,self.c2,
                                                                        self.a,self.b,
                                                                        self.r,self.d)