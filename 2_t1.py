# %%
import math
from tokenize import Double
import numpy as np
import cmath
import multiprocessing
from multiprocessing import Pool
import time


def JtoZ(r, theta):
    """ 极坐标转直角坐标

    Args:
        r (_type_): 极径
        theta (_type_): 角度

    Returns:
        _type_: [x, y]
    """
    theta = math.radians(theta)
    c = cmath.rect(r, theta)
    return np.array([c.real, c.imag]).reshape(1, 2)


def ZtoJ(x, y):
    """直角坐标转极坐标

    Args:
        x (_type_): x
        y (_type_): y

    Returns:
        _type_: [r, 角度]
    """
    c = complex(x, y)
    r, theta = cmath.polar(c)
    theta = math.degrees(theta)
    return (r, theta)


def getAngle(L1, L2):
    return math.acos(np.matmul(L1, L2.transpose()) / (np.linalg.norm(L1) * np.linalg.norm(L2) + np.spacing(1)))

# %% 单进程


# start = time.time()
# f_min = math.inf
# x_best, y_best = 0., 0.
# for x in np.linspace(-120, 120, 241*5):
#     for y in np.linspace(-120, 120, 241*5):
#         f_cur = f(x, y)
#         if f_cur < f_min:
#             f_min = f_cur
#             x_best = x
#             y_best = y

# print("x_best = %f, y_best = %f, f_min = %f" % (x_best, y_best, f_min))

# print("极坐标(%f, %f)" % (ZtoJ(x_best, y_best)))
# print(f'single-process time cost: {time.time() - start}s')
# %% 多进程并行
def f(x, y, P0, P1, P2, alpha_1, alpha_2, alpha_3):
    Pz = np.array([x, y]).reshape(1, 2)
    PzP0 = P0 - Pz
    PzP1 = P1 - Pz
    PzP2 = P2 - Pz
    return (getAngle(PzP0, PzP1) - alpha_1) ** 2 + (getAngle(PzP1, PzP2) - alpha_2) ** 2 + (getAngle(PzP0, PzP2) - alpha_3) ** 2


def getFmin(x_list, P0, P1, P2, alpha_1, alpha_2, alpha_3):
    f_min = math.inf
    x_best, y_best = 0., 0.
    for x in x_list:
        for y in np.linspace(-120, 120, 241*5):
            f_cur = f(x, y, P0, P1, P2, alpha_1, alpha_2, alpha_3)
            if f_cur < f_min:
                f_min = f_cur
                x_best = x
                y_best = y
    return (f_min, x_best, y_best)


def solve(drone1, drone2, drone3, pos_true):
    P0 = JtoZ(*drone1)
    P1 = JtoZ(*drone2)
    P2 = JtoZ(*drone3)
    P_true = JtoZ(*pos_true)
    PP0 = P0 - P_true
    PP1 = P1 - P_true
    PP2 = P2 - P_true

    alpha_1 = getAngle(PP0, PP1)
    alpha_2 = getAngle(PP1, PP2)
    alpha_3 = getAngle(PP0, PP2)

    n_core = 6
    total = 241*5
    each = total // n_core  # 每个进程负责的区间
    total_list = np.linspace(-120, 120, total)
    results = []
    start = time.time()
    pool = Pool(processes=n_core)
    for i in range(n_core - 1):
        result = pool.apply_async(
            getFmin, (total_list[i * each: (i+1)*each], P0, P1, P2, alpha_1, alpha_2, alpha_3,))
        results.append(result)
        print("process " + str(i) + " start at " +
              str(i*each) + " end at " + str((i+1)*each))
    result = pool.apply_async(
        getFmin, (total_list[(n_core-1)*each:], P0, P1, P2, alpha_1, alpha_2, alpha_3,))
    print("process " + str(n_core - 1) + " start at " +
          str((n_core-1)*each) + " end at " + str(len(total_list) - 1))
    results.append(result)
    pool.close()
    pool.join()
    f_min = math.inf
    x_best, y_best = 0., 0.
    for result in results:
        (f_cur, x, y) = result.get()
        if(f_cur < f_min):
            f_min = f_cur
            x_best = x
            y_best = y

    print("x_best = %f, y_best = %f, f_min = %f" % (x_best, y_best, f_min))

    print("极坐标(%f, %f)" % (ZtoJ(x_best, y_best)))
    print(f'multi-process time cost: {time.time() - start}s')
    return f_min, x_best, y_best


def T1():
    print("==================== T1 =======================")
    drone1 = (0, 0)
    drone2 = (100, 0)
    drone3 = (100, 40)
    pos_true = (105, 119.75)
    solve(drone1, drone2, drone3, pos_true)


def T2():
    print("==================== T2 =======================")
    drone1 = (0, 0)
    drone2 = (100, 0)
    pos_true = (112, 320.28)
    f_min = math.inf
    x_best, y_best = 0., 0.
    theta_best = 0
    for theta in range(0+40, 360, 40):
        drone3 = (100, theta)
        f_cur, x_cur, y_cur = solve(drone1, drone2, drone3, pos_true)
        if(f_cur < f_min):
            f_min = f_cur
            x_best = x_cur
            y_best = y_cur
            theta_best = theta
    print("x_best = %f, y_best = %f, f_min = %f, theta_best = %f" %
          (x_best, y_best, f_min, theta_best))

    print("极坐标(%f, %f)" % (ZtoJ(x_best, y_best)))


# %%
if __name__ == '__main__':
    # T1()
    T2()
