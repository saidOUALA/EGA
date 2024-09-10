import numpy as np

def Lorenz_63(S,t,sigma,rho,beta):
    """ Lorenz-63 dynamical model. """
    x_1 = sigma*(S[1]-S[0]);
    x_2 = S[0]*(rho-S[2])-S[1];
    x_3 = S[0]*S[1] - beta*S[2];
    dS  = np.array([x_1,x_2,x_3]);
    return dS

# define equation
def rossler_attractor(S,t, a=0.1, b=0.1, c=14):
    x_1 = -S[1]-S[2];
    x_2 = S[0]+ a*S[1];
    x_3 = b + S[2]*(S[0] - c);
    dS  = np.array([x_1,x_2,x_3]);
    return dS


def linear_osc(S,t,alpha):
    """ Lorenz-63 dynamical model. """
    x_1 = alpha*(S[1]);
    x_2 = -alpha*(S[0]);
    dS  = np.array([x_1,x_2]);
    return dS
