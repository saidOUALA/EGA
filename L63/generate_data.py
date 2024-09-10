import numpy as np
from scipy.integrate import odeint
from dynamical_models import Lorenz_63, rossler_attractor, linear_osc
def generate_data(GD):
    # use this to generate the same data for different simulations
    np.random.seed(1);

    if (GD.model == 'Lorenz_63'):
    
        # 5 time steps (to be in the attractor space)  
        x0 = np.array([8.0,0.0,30.0]);
        S = odeint(Lorenz_63,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        x0 = S[S.shape[0]-1,:];

        # generate true state (xt)
        S = odeint(Lorenz_63,x0,np.arange(0.01,GD.nb_loop_data+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));

    elif (GD.model == 'rossler_attractor'):
    
        # 5 time steps (to be in the attractor space)  
        x0 = np.array([8.0,0.0,30.0]);
        S  = odeint(rossler_attractor,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.a,GD.parameters.b,GD.parameters.c));
        x0 = S[S.shape[0]-1,:];

        # generate true state (xt)
        S = odeint(rossler_attractor,x0,np.arange(0.01,GD.nb_loop_data+0.000001,GD.dt_integration),args=(GD.parameters.a,GD.parameters.b,GD.parameters.c));

    if (GD.model == 'linear_osc'):
        # 5 time steps (to be in the attractor space)
        x0 = np.array([5.0, 0.0]);
        S = odeint(linear_osc, x0, np.arange(0, 5 + 0.000001, GD.dt_integration),
                   args=(GD.parameters.alpha,));
        x0 = S[S.shape[0] - 1, :];

        # generate true state (xt)
        S = odeint(linear_osc, x0, np.arange(0.01, GD.nb_loop_data + 0.000001, GD.dt_integration),
                   args=(GD.parameters.alpha,));

    # reinitialize random generator number
    np.random.seed()
    return S;