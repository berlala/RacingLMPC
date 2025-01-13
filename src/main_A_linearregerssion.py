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
    A, B, Error = Regression(xPID_cl, uPID_cl, lamb) #固定的所以是LTI
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

    # # ======================================================================================================================
    # # ========================================= PLOT TRACK =================================================================
    # # ======================================================================================================================

    print("===== Start Plotting")
    plotTrajectory(map, xPID_cl, xPID_cl_glob, uPID_cl, 'PID')
    plotTrajectory(map, xMPC_cl, xMPC_cl_glob, uMPC_cl, 'MPC')
    plt.show()


if __name__== "__main__":
  main()