from generate_data import generate_data
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import copy
from functorch import vmap, jacrev, make_functional_with_buffers, make_functional
from torchdiffeq import odeint as odeint_torch
from utils import reshape_dataset_to_torch, calc_loss, print_metrics
import time
seed = 4

# Generate Lorenz 63 simulations
class GD:
    model = 'Lorenz_63'
    class parameters:
        sigma = 10.0
        rho = 28.0
        beta = 8.0/3
    dt_integration = 0.01 # integration time
    nb_loop_data = 60.0

# run the data generation
dataset = generate_data(GD)

for seed in range(1):
    params = {}
    #closure mdl parameters
    params['dim_state']          = 3
    params['dim_output']          = 3
    params['transition_layers']  = 2
    params['dim_hidden_dyn_mdl']  = 3

    # learning params
    params['train_size']         = 200
    params['ntrain']             = 5
    params['dt_integration']     = GD.dt_integration
    params['pretrained']         = False
    params['Batch_size']         = 32
    params['seq_size']           = 10
    params['nb_part']            = 5
    params['output_folder']      = 'output_models/'
    params['model_save_file_name']          = 'L63_model_error_comp_graph_dopri5_Euler_seed_'+str(seed)+'.pt'
    params['device']             = 'cuda'
    params['seed']               = seed
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    X_train, Y_train, X_test, Y_test = reshape_dataset_to_torch(dataset,params['seq_size'], params['train_size'])

    training_dataset = torch.utils.data.TensorDataset(X_train,Y_train)
    val_dataset      = torch.utils.data.TensorDataset(X_test,Y_test)
    dataloaders = {
        'train': torch.utils.data.DataLoader(training_dataset, batch_size=params['Batch_size'], shuffle=True, pin_memory=False),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=params['Batch_size'], shuffle=False, pin_memory=False),
    }

    #Define the closure model in pytorch
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
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
    M_theta = closure_term(params).to(params['device'])

    class closure_mdl_dt(torch.nn.Module):
        def __init__(self, params):
            super(closure_mdl_dt, self).__init__()
            # dimensions : 
            self.dim_state = params['dim_state']
            self.closure = M_theta
            # params of the true L63 system
            sigma = np.array(GD.parameters.sigma)#np.random.uniform(size=(1))
            self.sigma = (torch.from_numpy(sigma).float())
            rho = np.array(GD.parameters.rho)#np.random.uniform(size=(1))
            self.rho = (torch.from_numpy(rho).float())
            beta = np.array(GD.parameters.beta)#np.random.uniform(size=(1))
            self.beta = (torch.from_numpy(beta).float())

        def Dyn_net(self, t, inp):
            arg = [self.sigma,self.rho,self.beta]
            grad = (torch.zeros((inp.size())).to(inp.device))
            grad[:,0] = arg[0]*(inp[:,1]-inp[:,0]);
            grad[:,1] = inp[:,0]*(arg[1]-inp[:,2])-inp[:,1];
            grad[:,2] = inp[:,0]*inp[:,1] #+  self.closure(inp)[:,-1]  #arg[2]*inp[:,2];
            return grad +self.closure(inp)
        def forward(self, inp, dt = 0.01, t0 = 0, mode_no_gd = False):# flow of the ODE, assuming the flow is autonomous so t0 is always 0
            pred = odeint_torch(self.Dyn_net, inp, torch.arange(0,dt+0.000001,dt).to(inp.device), method = 'rk4')
            return pred[-1,:,:]
    ddmdl_dt   = closure_mdl_dt(params).to(params['device'])
    class closure_mdl(torch.nn.Module):
        def __init__(self, params):
            super(closure_mdl, self).__init__()
            # dimensions : 
            self.model_dt = ddmdl_dt
        def forward(self,dt, n, x0):

            pred = [x0]
            for i in range(n):
                pred.append(self.model_dt(pred[-1],dt))
            pred_seq = torch.stack(pred)

            return pred_seq
    ddmdl   = closure_mdl(params)


    loss_train  = []
    loss_test  = []

    def train_model(model, optimizer, scheduler, device, num_epochs=25, dt = params['dt_integration'], seq_size = params['seq_size'], nb_part = params['nb_part']):
        try : 
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = 1e10
            memory_usage = []
            time_consumption = []
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
                memory_usage_per_epoch = []
                time_consumption_per_epoch = []
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
                            pred_mdl = model(dt, seq_size, x_data)
                            loss = calc_loss(pred_mdl, y_data, metrics)
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                #torch.cuda.synchronize()  # Wait for GPU operations to finish
                                #memory_before = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
                                start_time_batch = time.time()

                                loss.backward()

                                end_time_batch = time.time()
                                #torch.cuda.synchronize()
                                #memory_after = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0



                                optimizer.step()

                                #memory_usage_per_epoch.append(memory_before - memory_after)
                                time_consumption_per_epoch.append(end_time_batch - start_time_batch)
                                #print(memory_usage_per_epoch[-1])


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

                memory_usage.append(np.array(memory_usage_per_epoch))
                time_consumption.append(np.array(time_consumption_per_epoch))

            print('Best val loss: {:4f}'.format(best_loss))

            # load best model weights
            best_model_wts_ret = copy.deepcopy(model)
            best_model_wts_ret.load_state_dict(best_model_wts)

        except KeyboardInterrupt:
            print('Loading best model with respect to the validation error.')
            best_model_wts_ret = copy.deepcopy(model)
            best_model_wts_ret.load_state_dict(best_model_wts)
        return best_model_wts_ret,model,np.array(time_consumption)


    optimizer_ft     = torch.optim.Adam(ddmdl.parameters(), lr=0.01)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    model_best_valid, model,time_consumption = train_model(ddmdl, optimizer_ft, exp_lr_scheduler, device = params['device'], num_epochs=params['ntrain'],  dt = params['dt_integration'], seq_size = params['seq_size'], nb_part = params['nb_part'])
    torch.save(model.state_dict(), params['output_folder']+params['model_save_file_name'])

