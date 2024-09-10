from generate_data import generate_data
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import copy
from torchdiffeq import odeint as odeint_torch
from utils import reshape_dataset_to_torch, calc_loss, print_metrics
import time

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Generate Lorenz 63 simulation
class GD:
    model = 'Lorenz_63'

    class parameters:
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3

    dt_integration = 0.01  # integration time
    nb_loop_data = 60.0


# run the data generation
dataset = generate_data(GD)

# parameters of the training experiment
params = {'dim_state': 3, 'dim_output': 3, 'transition_layers': 2, 'dim_hidden_dyn_mdl': 3, 'train_size': 200,
          'ntrain': 5, 'dt_integration': GD.dt_integration, 'pretrained': False, 'Batch_size': 32, 'seq_size': 10,
          'nb_part': 5, 'output_folder': 'output_models/',
          'model_save_file_name': 'L63_exact_gradient.pt', 'device': 'cuda'}

# reshaping dataset
X_train, Y_train, X_test, Y_test = reshape_dataset_to_torch(dataset, params['seq_size'], params['train_size'])
training_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
val_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
dataloaders = {
    'train': torch.utils.data.DataLoader(training_dataset, batch_size=params['Batch_size'], shuffle=True,
                                         pin_memory=False),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=params['Batch_size'], shuffle=False, pin_memory=False),
}


class HybridMdl(torch.nn.Module):
    def __init__(self, params):
        super(HybridMdl, self).__init__()
        # params of the true L63 system
        self.sigma = GD.parameters.sigma
        self.rho = GD.parameters.rho
        self.beta = GD.parameters.beta

        # params of the neural network correction
        self.trans_layers = params['transition_layers']
        self.transLayers = torch.nn.ModuleList([torch.nn.Linear(params['dim_state'], params['dim_hidden_dyn_mdl'])])
        self.transLayers.extend([torch.nn.Linear(params['dim_hidden_dyn_mdl'], params['dim_hidden_dyn_mdl']) for i in
                                 range(1, params['transition_layers'])])
        self.out_transLayers = torch.nn.Linear(params['dim_hidden_dyn_mdl'], params['dim_state'])

    def closure(self, x):
        for i in range(self.trans_layers):
            x = torch.tanh(self.transLayers[i](x))
        x = self.out_transLayers(x)
        return x

    def Dyn_net(self, t, inp):
        grad = (torch.zeros((inp.size())).to(inp.device))
        grad[:, 0] = self.sigma * (inp[:, 1] - inp[:, 0])
        grad[:, 1] = inp[:, 0] * (self.rho - inp[:, 2]) - inp[:, 1]
        grad[:, 2] = inp[:, 0] * inp[:, 1]  # + self.beta*inp[:,2];
        return grad + self.closure(inp)

    def forward(self, inp, dt=0.01, t0=0,
                mode_no_gd=False):  # flow of the ODE, assuming the flow is autonomous so t0 is always 0
        pred = odeint_torch(self.Dyn_net, inp, torch.arange(0, dt + 0.000001, dt).to(inp.device), method='dopri5')
        return pred[-1, :, :]


hybrid_L63 = HybridMdl(params).to(params['device'])




def train_model(model, optimizer, scheduler, device, num_epochs=25, dt=params['dt_integration'],
                seq_size=params['seq_size']):
    try:
        loss_train = []
        loss_test = []
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            memory_usage_per_epoch = []
            time_consumption_per_epoch = []
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("LR", param_group['lr'])

                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0

                for x_data, y_data in dataloaders[phase]:
                    y_data = torch.swapaxes(y_data, 0, 1)
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
                            loss.backward()
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

        print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        best_model_wts_ret = copy.deepcopy(model)
        best_model_wts_ret.load_state_dict(best_model_wts)

    except KeyboardInterrupt:
        print('Loading best model with respect to the validation error.')
        best_model_wts_ret = copy.deepcopy(model)
        best_model_wts_ret.load_state_dict(best_model_wts)
    return best_model_wts_ret, model, loss_train, loss_test


optimizer_ft = torch.optim.Adam(hybrid_L63.parameters(), lr=0.01)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
model_best_valid, model, time_consumption = train_model(hybrid_L63, optimizer_ft, exp_lr_scheduler, device=params['device'],
                                                        num_epochs=params['ntrain'], dt=params['dt_integration'],
                                                        seq_size=params['seq_size'])
torch.save(model.state_dict(), params['output_folder'] + params['model_save_file_name'])
