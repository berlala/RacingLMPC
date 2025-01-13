import sys
sys.path.append('fnc/simulator')
sys.path.append('fnc/controller')
sys.path.append('fnc')
import matplotlib.pyplot as plt
from plot import plotTrajectory, plotClosedLoopLMPC, animation_xy, animation_states, saveGif_xyResults
from initControllerParameters import initMPCParams, initLMPCParams
from PredictiveControllers import MPC, LMPC, MPCParams
from PredictiveModel import PredictiveModel
from Utilities import Regression, PID
from SysModel import Simulator
from Track import Map
import numpy as np
import pickle
import pdb

def main():
    # ======================================================================================================================
    # ============================================= Initialize parameters  =================================================
    # ======================================================================================================================
    N = 14                                # Horizon length
    n = 6;                                # State, define in [2] eq(0)
    # 状态定义是[vx, vy, wz, e_theta, s, e_y]
    d = 2                                 # Input dimension
    x0 = np.array([0.5, 0, 0, 0, 0, 0])       # Initial condition 
    xS = [x0, x0]
    dt = 0.1

    map = Map(0.4)             # Initialize map， 0.4是赛道半宽
    vt = 0.8                   # target vevlocity

    # Initialize controller parameters
    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt) # 参数是一样的
    
    # Init simulators
    simulator     = Simulator(map)

    # ======================================================================================================================
    # ======================================= PID path following ===========================================================
    # ======================================================================================================================
    print("Starting PID")
    # Initialize pid and run sim
    PIDController = PID(vt)
    xPID_cl, uPID_cl, xPID_cl_glob, _ = simulator.sim(xS, PIDController)
    print("===== PID terminated")

    # ======================================================================================================================
    # ======================================  LINEAR REGRESSION ============================================================
    # ======================================================================================================================
    print("Starting LINEAR REGRESSION")
    # Estimate system dynamics
    lamb = 0.0000001
    A, B, Error = Regression(xPID_cl, uPID_cl, lamb) 
    #固定的所以是LTI, 等效于一个离散系统，但是其辨识对工况稳定要求很高
    mpcParam.A = A  # 6*6
    print("A is ")  
    print(A)
    mpcParam.B = B  # 6*2
    print("B is ")
    print(B)
    # Initialize MPC and run closed-loop sim
    mpc = MPC(mpcParam)
    xMPC_cl, uMPC_cl, xMPC_cl_glob, _ = simulator.sim(xS, mpc)
    print("===== LINEAR REGRESSION terminated")

    # ======================================================================================================================
    # ======================================  Model Base Method ============================================================
    # ======================================================================================================================
    print("Starting Model Base Method")
    # Estimate system dynamics

    #轮胎力模型参数
    m  = 1.98    
    lf = 0.125   
    lr = 0.125   
    Iz = 0.024   
    Df = 0.8 * m * 9.81 / 2.0  
    Dr = 0.8 * m * 9.81 / 2.0  
    Cf = 1.25    
    Bf = 1.0     
    Cr = 1.25    
    Br = 1.0     

    # 在alpha=0处的导数
    # 前轮: Fyf = Df * sin(Cf * arctan(Bf * alpha_f))
    # 后轮: Fyr = Dr * sin(Cr * arctan(Br * alpha_r))

    # 使用链式法则计算导数
    # d(sin(arctan(x)))/dx = 1/(1 + x^2) at x=0
    dFyf_dalpha = Df * Cf * Bf  # 在alpha_f = 0处
    dFyr_dalpha = Dr * Cr * Br  # 在alpha_r = 0处

    # 或者更准确的表达式（考虑实际工作点）
    # alpha_f = delta - arctan((vy + lf*wz)/vx)  # 前轮侧偏角
    # alpha_r = -arctan((vy - lr*wz)/vx)         # 后轮侧偏角

    # 在实际工作点处的导数
    alpha_f = 0  # 可以用实际值替换
    alpha_r = 0  # 可以用实际值替换

    dFyf_dalpha = Df * Cf * Bf / (1 + (Bf * alpha_f)**2)
    dFyr_dalpha = Dr * Cr * Br / (1 + (Br * alpha_r)**2)
    
    wz = x0[2]
    vy = x0[1]
    vx = x0[0]
    v0 = vt
    # 构建A矩阵 (6x6) [vx, vy, wz, epsi, s, ey]
    A = np.zeros((6, 6))
    # 纵向动力学: v̇x = a + vy*wz
    A[0,1] = wz   # 来自vy的影响
    A[0,2] = vy   # 来自wz的影响

    # 横向动力学: v̇y = (Fyf + Fyr)/m - vx*wz
    A[1,0] = -wz  # 来自vx的影响
    A[1,1] = -(dFyf_dalpha + dFyr_dalpha)/(m*v0)
    A[1,2] = -vx + (dFyr_dalpha*lr - dFyf_dalpha*lf)/(m*v0)

    # 横摆动力学: ẇz = (Fyf*lf - Fyr*lr)/Iz
    A[2,1] = (dFyr_dalpha*lr - dFyf_dalpha*lf)/(Iz*v0)
    A[2,2] = -(dFyf_dalpha*lf*lf + dFyr_dalpha*lr*lr)/(Iz*v0)

    # 运动学关系保持不变
    A[3,2] = 1    # e_theta导数等于wz
    A[4,0] = 1    # s导数等于vx
    A[5,1] = 1    # ey导数包含vy
    A[5,3] = vx   # 使用实际vx而不是v0

    # B矩阵保持不变，因为输入的影响是直接的
    B = np.zeros((6, 2))
    B[1,0] = dFyf_dalpha/m
    B[2,0] = dFyf_dalpha*lf/Iz
    B[0,1] = 1.0

    # Discrete time model
    Ad = np.eye(6) + A * dt  # 一阶欧拉法
    Bd = B * dt

    mpcParam.A = Ad
    mpcParam.B = Bd
    # Initialize MPC and run closed-loop sim
    print("Model A is ")  
    print(A)
    print("Model B is ")
    print(B)
    mpc = MPC(mpcParam)
    xMPC_model, uMPC_model, xMPC_model_glob, _ = simulator.sim(xS, mpc)
    print("===== Model-based terminated")

    # # ======================================================================================================================
    # # ========================================= PLOT TRACK =================================================================
    # # ======================================================================================================================

    print("===== Start Plotting")
    plotTrajectory(map, xPID_cl, xPID_cl_glob, uPID_cl, 'PID')
    plotTrajectory(map, xMPC_cl, xMPC_cl_glob, uMPC_cl, 'MPC')
    plotTrajectory(map, xMPC_model, xMPC_model_glob, uMPC_model, 'Model Base')
    plt.show()


if __name__== "__main__":
  main()