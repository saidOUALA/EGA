import sys
sys.path.append('/scratch/so2495/code/EnBack/2T_L96/generate_data')
from generate_data import generate_data
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import time
import copy
from functorch import vmap, jacrev, make_functional_with_buffers, make_functional
from torchdiffeq import odeint as odeint_torch
from scipy.integrate import odeint
seed = 4


# In[2]:
def reshape_dataset_to_torch(dataset, time_steps, train_size):
    data_X, data_Y = [], []
    for i in range(len(dataset)-time_steps):
        a = dataset[i,:]
        data_X.append(a)
        data_Y.append(dataset[i:i+time_steps+1,:])
    X_train = torch.from_numpy(np.array(data_X)[:train_size]).float()
    Y_train = torch.from_numpy(np.array(data_Y)[:train_size]).float()
    X_test  = torch.from_numpy(np.array(data_X)[train_size:]).float()
    Y_test  = torch.from_numpy(np.array(data_Y)[train_size:]).float()
    return X_train, Y_train, X_test, Y_test
def calc_loss(pred, target, metrics):
    loss_mse = torch.mean((pred - target) ** 2)
    loss = loss_mse
    metrics['loss'] += loss_mse.data.cpu().numpy() * target.size(0)
    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))


# # Generate training data : 

# In[3]:


# Generate Lorenz 63 simulations
class GD:
    model = 'Lorenz_63'
    class parameters:
        sigma = 10.0
        rho = 28.0
        beta = 8.0/3
    dt_integration = 0.2 # integration time
    dt_states = 1 # number of integeration times between consecutive states (for xt and catalog)
    dt_obs = 8# number of integration times between consecutive observations (for yo)
    var_obs = np.array([0]) # indices of the observed variables
    nb_loop_train = 60.01 # size of the catalog
    nb_loop_test = 200 # size of the true state and noisy observations
    sigma2_catalog = 0.0 # variance of the model error to generate the catalog
    sigma2_obs = 2.0 # variance of the observation error to generate observation
    
# run the data generation
catalog, xt, yo = generate_data(GD)


# In[4]:

for seed in range(5):
    params = {}
    #closure mdl parameters
    params['dim_state']          = 3
    params['dim_output']          = 3
    params['transition_layers']  = 2
    params['dim_hidden_dyn_mdl']  = 3

    # learning params
    params['train_size']         = 200
    params['ntrain']             = [30000]
    params['dt_integration']     = GD.dt_integration
    params['pretrained']         = False
    params['Batch_size']         = 32
    params['seq_size']           = 10
    params['nb_part']            = 5
    params['nb_batch']           = int(catalog.analogs.shape[0]/params['Batch_size'])
    params['output_folder']      = 'output_models/'
    params['model_save_file_name']          = 'L63_model_error_ensemble_jacobians_dopri5_Euler_dt02_seed_'+str(seed)+'.pt'
    params['device']             = 'cpu'
    params['seed']               = seed
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])


    # In[5]:


    X_train, Y_train, X_test, Y_test = reshape_dataset_to_torch(catalog.analogs,params['seq_size'], params['train_size'])


    # In[6]:


    training_dataset = torch.utils.data.TensorDataset(X_train,Y_train) # create your datset
    val_dataset      = torch.utils.data.TensorDataset(X_test,Y_test)
    dataloaders = {
        'train': torch.utils.data.DataLoader(training_dataset, batch_size=params['Batch_size'], shuffle=True, pin_memory=False),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=params['Batch_size'], shuffle=False, pin_memory=False),
    }       


    # # Define the closure model in pytorch 

    # In[7]:


    class closure_term(torch.nn.Module):
        def __init__(self, params):
            super(closure_term, self).__init__()
            #params of the approximate dynamical model (bilinear model)
            self.trans_layers = params['transition_layers']
            self.transLayers  = torch.nn.ModuleList([torch.nn.Linear(params['dim_state'], params['dim_hidden_dyn_mdl'])])
            self.transLayers.extend([torch.nn.Linear(params['dim_hidden_dyn_mdl'], params['dim_hidden_dyn_mdl']) for i in range(1, params['transition_layers'])])
            self.out_transLayers = torch.nn.Linear(params['dim_hidden_dyn_mdl'],params['dim_state']) 
        def forward(self, inp):
            aug_vect = inp#torch.cat((L_outp, BP_outp), dim=1)
            #print(aug_vect)
            for i in range((self.trans_layers)):
                aug_vect = torch.tanh(self.transLayers[i](aug_vect))
            aug_vect = self.out_transLayers(aug_vect)
            grad = aug_vect#self.outputLayer(aug_vect)
            return grad 
    M_theta = closure_term(params)


    # # Define the numpy model, with the closure term : 

    # In[8]:


    def AnDA_Lorenz_63(S,t,model,sigma = GD.parameters.sigma,rho = GD.parameters.rho,beta = GD.parameters.beta):
        """ Lorenz-63 dynamical model. """
        M = model(torch.from_numpy(S).float().unsqueeze(0)).squeeze(0).detach().numpy()
        dS = np.zeros_like(S)
        dS[0] = sigma*(S[1]-S[0]);
        dS[1] = S[0]*(rho-S[2])-S[1];
        dS[2] = S[0]*S[1]# + M[-1]
        return dS + M
    def time_stepper(x0,dt,model):
        S = odeint(AnDA_Lorenz_63,x0,np.arange(0,GD.dt_integration+0.000001,GD.dt_integration),args=(model,));
        return S[-1,:]#pred#
    def forward_black_box_model(model,dt, n, x0, nb_part, sigma_noise = 1.0):
        # add an ensemble dimension to x0 : 
        x0  = x0.unsqueeze(1)# add a dimension as a channel (for the ensemble size)
        x0  = x0.repeat(1,nb_part, 1)

        x0_init = (x0.clone())

        eps  = torch.randn(x0.shape[0], nb_part-1, x0.shape[-1])*sigma_noise
        x0_init[:,1:,:] = x0_init[:,1:,:] + eps

        pred    = [x0]
        pred_x0 = [x0_init]
        comp_to = [x0[:-n]]
        for i in range(n):# loop over sequence
            #print(i)
            # generate the noise that will be applied on the ensemble members

            # always take the initial condition as the perturbed previous forecast, not the perturbed predictions
            # so that the ensemble does not spread too much
            #pred_x0[-1][:,1:,:] = pred[-1][:,:1,:].repeat(1,nb_part-1, 1) + eps
            pred_batch = torch.zeros_like(pred[-1])
            for k in range(x0.shape[0]):# loop over batch 
                for j in range(nb_part):# loop over ensemble 
                    pred_batch[k,j,:] = torch.from_numpy(time_stepper(pred_x0[-1][k,j,:].detach().numpy(),dt,model))

            eps  = torch.randn(x0.shape[0], nb_part-1, x0.shape[-1])*sigma_noise
            pred_batch_x0 = (pred_batch.clone())
            pred_batch_x0[:,1:,:] = pred_batch_x0[:,:1,:].repeat(1,nb_part-1, 1) + eps

            pred.append(pred_batch)
            pred_x0.append(pred_batch_x0)
            if -n+(i+1) == 0:
                comp_to.append(x0[i+1:])
            else:
                comp_to.append(x0[i+1:-n+(i+1)])
        pred_seq = torch.stack(pred)#[:,:,:]
        pred_x0  = torch.stack(pred_x0)#[:,:,:]

        return pred_seq[:,:,0,:], pred_seq-pred_seq.mean(dim = -2).unsqueeze(-2), pred_x0 - pred_x0.mean(dim = -2).unsqueeze(-2)
    def init_grads_to_zero(model,x):
        tmp = model(x).sum()
        tmp.backward()
        model.zero_grad()
    def compute_and_set_gradients(model,pred_mdl,dt, grad0, jacs = None):
        n_steps, bs, dim = pred_mdl.size()
        func, theta            = make_functional(model)
        jac_theta_dict         = vmap(jacrev(func), (None, 0))(theta, pred_mdl.detach())
        jac_theta_dict_flatten = [j.reshape(n_steps, bs, dim, -1) for j in jac_theta_dict]
        jac_theta_dict_flatten = torch.cat(jac_theta_dict_flatten,dim = -1)*dt

        grad_phi_n_all = []
        for kk in range(jac_theta_dict_flatten.shape[-1]):
                grad_phi_n = []
                for m in range(0,n_steps-1):
                    sum_cont = 0
                    for k in range(m+1):
                        tmp = torch.eye(dim)
                        tmp = tmp.reshape((1, dim, dim))
                        tmp = tmp.repeat(pred_mdl.shape[1], 1, 1)
                        running_jac = [tmp]
                        for i in range(m,k,-1):
                            running_jac.append(torch.bmm(running_jac[-1],jacs[i,:,:,:]))
                        sum_cont += torch.bmm(running_jac[-1],jac_theta_dict_flatten[k,:,:,kk:kk+1])
                    grad_phi_n.append(sum_cont)
                grad_phi_n = torch.stack(grad_phi_n)
                grad_phi_n_all.append(grad_phi_n)

        grad_theta_n_all = []
        for kk in range(len(grad_phi_n_all)):
            grad_theta_n = []
            for i in range(n_steps-1):
                grad_theta_n.append(torch.bmm(grad0.unsqueeze(2)[1+i],grad_phi_n_all[kk][i]))
            grad_theta_n = torch.stack(grad_theta_n)
            grad_theta_n_all.append(grad_theta_n)
        #grad_theta_n_all = torch.stack(grad_theta_n_all)

        for name, param in model.named_parameters():
            shape_all = param.grad.reshape(-1).shape[0]
            #elem = torch.arange(shape_all).int()
            param.grad.data = torch.stack(grad_theta_n_all[:shape_all]).sum(dim =1).sum(dim =1).reshape(param.grad.data.shape)
            del grad_theta_n_all[:shape_all]
        return grad_theta_n_all
    def compute_jacobian_approximation(pred_mdl_x0, pred_mdl):
        n_steps, batch_size, nb_part, dim = pred_mdl_x0.size()
        jacs = torch.zeros((n_steps,batch_size,dim, dim))
        for i in range(n_steps-1):
            for j in range(pred_mdl.shape[1]):
                yxt = torch.mm(pred_mdl[i+1,j,:].T,pred_mdl_x0[i,j,:])
                xxt = torch.mm(pred_mdl_x0[i,j,:].T,pred_mdl_x0[i,j,:])
                A   = torch.mm(yxt,torch.inverse(xxt))
                jacs[i,j,:,:] = A
        #jacs = torch.from_numpy(jacs).float()
        return jacs
    def store_params(model,grad_list,params_list):
        params_wg = []
        grads_wg = []
        grads_ng = []
        for name, param in model.named_parameters():
            if not (param.grad == None):
                grads_wg.append(param.grad.reshape(-1).detach().clone())
                params_wg.append(param.reshape(-1).detach().clone())
        grads_wg = torch.cat(grads_wg)
        params_wg = torch.cat(params_wg)
        grad_list.append(grads_wg)
        params_list.append(params_wg)
        return grad_list, params_list


    # In[9]:


    loss_train  = []
    loss_test  = []
    grad_list, param_list = [], []
    def train_model(model, optimizer, scheduler, device, num_epochs=25, dt = params['dt_integration'], seq_size = params['seq_size'], nb_part = params['nb_part'], grad_list = grad_list, param_list = param_list):
        try : 
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = 1e10

            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
                since = time.time()

                # Each epoch has a training and validation phase
                for phase in ['train','val']:
                    if phase == 'train':
                        scheduler.step()
                        for param_group in optimizer.param_groups:
                            print("LR", param_group['lr'])

                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    metrics = defaultdict(float)
                    epoch_samples = 0

                    for x_data, y_data in dataloaders[phase]:
                        y_data = torch.swapaxes(y_data,0,1)
                        x_data = x_data.to(device)
                        y_data = y_data.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        training_mode = phase == 'train'
                        with torch.set_grad_enabled(training_mode):
        #                    print('max inps :' ,inputs.max())
                            pred_mdl, pred_mdl_xdt, pred_mdl_x0 = forward_black_box_model(model, dt, seq_size, x_data, nb_part = nb_part)
                            loss = calc_loss(pred_mdl, y_data, metrics)
                            # backward + optimize only if in training phase
                            if phase == 'train':

                                jacs   = compute_jacobian_approximation(pred_mdl_x0, pred_mdl_xdt)
                                coef   = pred_mdl.reshape(-1).shape[0]
                                grad0  = (2*(pred_mdl - y_data))/coef
                                gradss = compute_and_set_gradients(model,pred_mdl,dt,grad0, jacs = jacs)
                                grad_list, params_list = store_params(model,grad_list,param_list)
                                optimizer.step()

                        # statistics
                        epoch_samples += x_data.size(0)

                    print_metrics(metrics, epoch_samples, phase)
                    epoch_loss = metrics['loss'] / epoch_samples
                    if phase == 'train':
                        loss_train.append(epoch_loss)
                    # deep copy the model
                    if phase == 'val' and epoch_loss < best_loss:
                        loss_test.append(epoch_loss)
                        print("saving best model")
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())

                time_elapsed = time.time() - since
                print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            print('Best val loss: {:4f}'.format(best_loss))

            # load best model weights
            best_model_wts_ret = copy.deepcopy(model)
            best_model_wts_ret.load_state_dict(best_model_wts)

        except KeyboardInterrupt:
            print('Loading best model with respect to the validation error.')
            best_model_wts_ret = copy.deepcopy(model)
            best_model_wts_ret.load_state_dict(best_model_wts)
        return best_model_wts_ret,model, grad_list, param_list


    # In[ ]:


    optimizer_ft     = torch.optim.Adam(M_theta.parameters(), lr=0.01)
    init_grads_to_zero(M_theta, X_train[:2])
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    model_best_valid, model, grad_list, param_list = train_model(M_theta, optimizer_ft, exp_lr_scheduler, device = params['device'], num_epochs=300,  dt = params['dt_integration'], seq_size = params['seq_size'], nb_part = params['nb_part'], grad_list = grad_list, param_list = param_list)
    torch.save(model.state_dict(), params['output_folder']+params['model_save_file_name'])


# In[ ]:




