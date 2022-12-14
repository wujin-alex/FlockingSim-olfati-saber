import numpy as np

def sigma_norm(z, epsilon):
    """
    Eq (8)
    """
    if isinstance(z, (int, float)):
        return (np.sqrt(1 + epsilon * z**2) - 1) / epsilon
    elif isinstance(z, np.ndarray):
        if z.ndim == 1:
            if z.shape[0] > 3:
                return (np.sqrt(1 + epsilon * z**2) - 1) / epsilon
            else:
                return (np.sqrt(1 + epsilon * np.linalg.norm(z)**2) - 1) / epsilon
        elif z.ndim == 2:
            return (np.sqrt(1 + epsilon * np.linalg.norm(z, axis=0)**2) - 1) / epsilon
        else:
            return None

def sigma_norm_gradient(z, epsilon):
    """
    Eq (9)
    """
    if isinstance(z, (int, float)):
        return z / np.sqrt(1 + epsilon * z**2)
    elif isinstance(z, np.ndarray):
        if z.ndim == 1:
            ## 如果维度大小大于3，认为是多个点，否则认为是维度为2/3的单点
            if z.shape[0] > 3:
                return z / np.sqrt(1 + epsilon * z**2)
            else:
                return z / np.sqrt(1 + epsilon * np.linalg.norm(z)**2)
        elif z.ndim == 2:
            return z / np.sqrt(1 + epsilon * np.linalg.norm(z, axis=0)**2)
        else:
            return None

def bump_function(z, h):
    """
    Eq (10)
    """
    if isinstance(z, (int, float)):
        if z >= 0 and z < h:
            return 1.0
        elif z >=h and z <= 1:
            return (1 + np.cos(np.pi*(z-h)/(1-h))) / 2.0
        else:
            return 0
    elif isinstance(z, np.ndarray):
        res = np.zeros(z.shape)
        mask = (z>=0) & (z<h)
        res[mask] = 1.0
        mask = (z>=h) & (z<=1)
        res[mask] = (1 + np.cos(np.pi*(z[mask]-h)/(1-h))) / 2.0
        return res

def aij(qi, qj, epsilon, r, h):
    """
    Eq (11)
    """
    q_ji    = sigma_norm(np.linalg.norm(qj - qi), epsilon)
    r_alpha = sigma_norm(r, epsilon)

    return bump_function(q_ji / r_alpha, h)

def nij(qi, qj, epsilon):
    """
    Eq (25)
    """
    q_ji = qj - qi
    res = sigma_norm_gradient(q_ji, epsilon)
    return res

def uneven_sigmoidal_fucntion(z, a, b):
    """
    Eq (15)
    """
    if a < 0 or b < a:
        raise(ValueError("a,b must satify: 0<a<=b"))
    c = np.abs(a-b) / np.sqrt(4*a*b)

    res = ((a + b) * sigma_norm_gradient(z+c, 1) + (a - b)) / 2.0
    return res
    
def action_function(z, epsilon, r, h, d, a, b):
    """
    Eq (15)
    """
    r_alpha = sigma_norm(r, epsilon)
    d_alpha = sigma_norm(d, epsilon)

    res = bump_function(z/r_alpha, h) * uneven_sigmoidal_fucntion(z-d_alpha, a, b)
    return res

def bik(qi, qk, epsilon, h, d_obs):
    """
    Eq (64)
    """
    q_sigma = sigma_norm(qk - qi, epsilon)
    d_sigma = sigma_norm(d_obs, epsilon)
    return bump_function(q_sigma/d_sigma, h)

def repulsive_action_function(z, epsilon, d_obs, h):
    """
    Eq (65)
    """
    d_sigma = sigma_norm(d_obs, epsilon)
    res = bump_function(z/d_sigma, h) * (sigma_norm_gradient(z-d_sigma, 1) - 1)
    return res

def get_beta_agent_hyperplane(qi, pi):
    pass

def  get_beta_agent_sphere(qi, pi, y, r):
    """
    qi: a-agent位置
    pi: a-agent速度
    y: 障碍物圆心坐标
    r: 障碍物半径
    """
    u = r / np.linalg.norm(qi-y)
    a = (qi-y) / np.linalg.norm(qi-y)
    a = a.reshape(-1, 1)
    P = np.eye(qi.shape[0]) - np.dot(a, a.T)

    qk = u*qi + (1-u)*y
    pk = u*np.dot(P,pi.reshape(-1,1))
    pk = pk.reshape(-1)

    return qk,pk