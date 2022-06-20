import random
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Hardware constants
F = 6.0 * 9.81 # average thrust N
M = 5.0
I = 0.3 # mass moment of inertia kg.m^2
D = 0.45 # distance from TVC to flight computer m
DT = 0.001 # delta time s

# Graphin place-holders
GraphX = []
GraphY = []

# State space beginning values and setpoint
theta = 0.0 # angle of the rocket rad
theta_dot = 0.07 # velocity of the rocket rad
z = 0.0 # horizontal translation of the rocket m
z_dot = 0.0 # horizontal velocity of the rocket m/s

# State Space matrices
# A = np.matrix([[0, 1],
#                [0, 0]]) # constant state matrix

A = np.matrix([[0,     1, 0, 0],
               [0,     0, 0, 0],
               [0,     0, 0, 1],
               [F / M, 0, 0, 0]])

# B = np.matrix([[0],
#                [F * D / I]]) # constant input matrix

B = np.matrix([[0],
               [1 / I],
               [0],
               [-1 / (M * D)]])

# H infinity controller???

Q = np.matrix([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]]) # "stabalise the system"

R = np.matrix([[1]]) # "cost of energy to the system"

x = np.matrix([[theta],
               [theta_dot],
               [z],
               [z_dot]]) # state vector matrix

xf = np.matrix([[0],
                [0],
                [0],
                [0]]) # setpoint matrix

# LQR function
def Lqr(A,B,Q,R):
    # solves algebraic riccati equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    # computes the optimal K value
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    # compute the eigenvalues
    S = np.linalg.eigvals(A-np.dot(B,K))
    return K, X, S

# State function
def UpdateState(A, x, B, u):
    return 

if __name__ == "__main__":
    # main loop
    for t in range(2000):
        # gets optimal gain (K)
        K, S, E = Lqr(A, B, Q, R)
        # calculates the error from setpoint
        e = x - xf
        # calculates optimal output (u)
        u = -K * e
        # updates the state (x)
        x += DT * (A * x + B * u)
        # appending the graph
        GraphX.append(t * DT)
        GraphY.append(float(x[1]) * 180 / np.pi)

    # outputs optimal gain
    print(K)

    # plots the matplotlib graph
    plt.plot(GraphX, GraphY)
    plt.xlabel('time (s)')
    plt.ylabel('output (deg)')
    plt.title('State Space Output')
    plt.show()