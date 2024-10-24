import torch
import numpy as np
import copy
from scipy.integrate import odeint
from collections import defaultdict
from functorch import vmap, jacrev, make_functional
import time

def reshape_dataset_to_torch(dataset, time_steps, train_size):
    data_X, data_Y = [], []
    for i in range(len(dataset) - time_steps):
        a = dataset[i, :]
        data_X.append(a)
        data_Y.append(dataset[i:i + time_steps + 1, :])
    X_train = torch.from_numpy(np.array(data_X)[:train_size]).float()
    Y_train = torch.from_numpy(np.array(data_Y)[:train_size]).float()
    X_test = torch.from_numpy(np.array(data_X)[train_size:]).float()
    Y_test = torch.from_numpy(np.array(data_Y)[train_size:]).float()
    return X_train, Y_train, X_test, Y_test



def extract_and_reset_grads_norm_grads(model):
    grads = []
    norm_grads = []
    for name, param in model.named_parameters():
        if not (param.grad is None):
            grads.append(param.grad.reshape(-1).detach().clone())
            # Compute the norm of the weights
            weight_norm = torch.norm(param.data)

            # Compute the norm of the gradients
            grad_norm = torch.norm(param.grad)
            if weight_norm != 0:  # Avoid division by zero
                norm_grads.append(param.grad / weight_norm)
            else:
                print("weight_norm is zero for "+ name +", returning nan")
                norm_grads.append(np.nan)



    grads = torch.cat(grads)
    norm_grads = torch.cat(norm_grads)
    # reset params and gradients of EGA ST
    model.zero_grad()
    return grads, norm_grads


def extract_and_reset_grads(model):
    grads = []
    for name, param in model.named_parameters():
        if not (param.grad is None):
            grads.append(param.grad.reshape(-1).detach().clone())
    grads = torch.cat(grads)
    model.zero_grad()
    return grads


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


def train_L63(model, dataloaders, optimizer, scheduler, device, num_epochs, dt, seq_size, grad_mode):
    try:
        loss_train = []
        loss_val = []
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
                        pred_mdl = model(dt, seq_size, x_data, grad_mode = grad_mode)
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
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("LR", param_group['lr'])
                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    loss_val.append(epoch_loss)
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
    return best_model_wts_ret, model, loss_train, loss_val


def Lorenz_63(S, t, model, parameters, closure=True):
    """ Lorenz-63 dynamical model. """
    M = model(torch.from_numpy(S).float().unsqueeze(0)).squeeze(0).cpu().detach().numpy()
    dS = np.zeros_like(S)
    dS[0] = parameters.sigma * (S[1] - S[0]);
    dS[1] = S[0] * (parameters.rho - S[2]) - S[1];
    dS[2] = S[0] * S[1]
    if closure:
        return dS + M
    else:
        return dS


def time_stepper(x0, dt, model, parameters):
    S = odeint(Lorenz_63, x0, np.arange(0, dt + 0.000001, dt), args=(model, parameters));
    return S[-1, :]


def forward_black_box_model(model, dt, n, x0, nb_part, parameters, sigma_noise=1.0):
    # add an ensemble dimension to x0 :
    x0 = x0.unsqueeze(1)  # add a dimension as a channel (for the ensemble size)
    x0 = x0.repeat(1, nb_part, 1)

    x0_init = (x0.clone())

    eps = torch.randn(x0.shape[0], nb_part - 1, x0.shape[-1]).to(x0.device) * sigma_noise
    x0_init[:, 1:, :] = x0_init[:, 1:, :] + eps

    pred = [x0]
    pred_x0 = [x0_init]
    comp_to = [x0[:-n]]
    for i in range(n):  # loop over sequence
        pred_batch = torch.zeros_like(pred[-1]).to(x0.device)
        for k in range(x0.shape[0]):  # loop over batch
            for j in range(nb_part):  # loop over ensemble
                pred_batch[k, j, :] = torch.from_numpy(
                    time_stepper(pred_x0[-1][k, j, :].cpu().detach().numpy(), dt, model, parameters)).to(x0.device)

        eps = torch.randn(x0.shape[0], nb_part - 1, x0.shape[-1]).to(x0.device) * sigma_noise
        pred_batch_x0 = (pred_batch.clone())
        pred_batch_x0[:, 1:, :] = pred_batch_x0[:, :1, :].repeat(1, nb_part - 1, 1) + eps

        pred.append(pred_batch)
        pred_x0.append(pred_batch_x0)
        if -n + (i + 1) == 0:
            comp_to.append(x0[i + 1:])
        else:
            comp_to.append(x0[i + 1:-n + (i + 1)])
    pred_seq = torch.stack(pred)
    pred_x0 = torch.stack(pred_x0)
    return pred_seq[:, :, 0, :], pred_seq - pred_seq.mean(dim=-2).unsqueeze(-2), pred_x0 - pred_x0.mean(
        dim=-2).unsqueeze(-2)


def init_grads_to_zero(model, x):
    tmp = model(x).sum()
    tmp.backward()
    model.zero_grad()


def compute_and_set_gradients(model, pred_mdl, dt, grad0, jacs=None):
    n_steps, bs, dim = pred_mdl.size()
    func, theta = make_functional(model)
    jac_theta_dict = vmap(jacrev(func), (None, 0))(theta, pred_mdl.detach())
    jac_theta_dict_flatten = [j.reshape(n_steps, bs, dim, -1) for j in jac_theta_dict]
    jac_theta_dict_flatten = torch.cat(jac_theta_dict_flatten, dim=-1) * dt

    grad_phi_n_all = []
    for kk in range(jac_theta_dict_flatten.shape[-1]):
        grad_phi_n = []
        for m in range(0, n_steps - 1):
            sum_cont = 0
            for k in range(m + 1):
                tmp = torch.eye(dim)
                tmp = tmp.reshape((1, dim, dim))
                tmp = tmp.repeat(pred_mdl.shape[1], 1, 1)
                running_jac = [tmp]
                for i in range(m, k, -1):
                    running_jac.append(torch.bmm(running_jac[-1], jacs[i, :, :, :]))
                sum_cont += torch.bmm(running_jac[-1], jac_theta_dict_flatten[k, :, :, kk:kk + 1])
            grad_phi_n.append(sum_cont)
        grad_phi_n = torch.stack(grad_phi_n)
        grad_phi_n_all.append(grad_phi_n)

    grad_theta_n_all = []
    for kk in range(len(grad_phi_n_all)):
        grad_theta_n = []
        for i in range(n_steps - 1):
            grad_theta_n.append(torch.bmm(grad0.unsqueeze(2)[1 + i], grad_phi_n_all[kk][i]))
        grad_theta_n = torch.stack(grad_theta_n)
        grad_theta_n_all.append(grad_theta_n)
    # grad_theta_n_all = torch.stack(grad_theta_n_all)

    for name, param in model.named_parameters():
        shape_all = param.reshape(-1).shape[0]
        param.grad.data = torch.stack(grad_theta_n_all[:shape_all]).sum(dim=1).sum(dim=1).reshape(param.shape)
        del grad_theta_n_all[:shape_all]
    return grad_theta_n_all


def compute_jacobian_approximation(pred_mdl_x0, pred_mdl):
    n_steps, batch_size, nb_part, dim = pred_mdl_x0.size()
    jacs = torch.zeros((n_steps, batch_size, dim, dim))
    for i in range(n_steps - 1):
        for j in range(pred_mdl.shape[1]):
            yxt = torch.mm(pred_mdl[i + 1, j, :].T, pred_mdl_x0[i, j, :])
            xxt = torch.mm(pred_mdl_x0[i, j, :].T, pred_mdl_x0[i, j, :])
            A = torch.mm(yxt, torch.inverse(xxt))
            jacs[i, j, :, :] = A
    # jacs = torch.from_numpy(jacs).float()
    return jacs


def store_params(model, grad_list, params_list):
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

def train_L63_EGA_Ensemble(model, dataloaders, optimizer, scheduler, device, num_epochs, dt, seq_size, nb_part, parameters):
    try :
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is None:
                    # Initialize grad tensor if it doesn't exist
                    param.grad = torch.zeros_like(param)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10
        loss_train, loss_val = [], []

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            since = time.time()
            # Each epoch has a training and validation phase
            for phase in ['train','val']:
                if phase == 'train':
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
                    optimizer.zero_grad(set_to_none = False)

                    # forward
                    # track history if only in train
                    training_mode = phase == 'train'
                    with torch.set_grad_enabled(training_mode):
                        pred_mdl, pred_mdl_xdt, pred_mdl_x0 = forward_black_box_model(model, dt, seq_size, x_data, nb_part = nb_part, parameters = parameters)
                        loss = calc_loss(pred_mdl, y_data, metrics)
                        # backward + optimize only if in training phase
                        if phase == 'train':

                            jacs   = compute_jacobian_approximation(pred_mdl_x0, pred_mdl_xdt)
                            coef   = pred_mdl.reshape(-1).shape[0]
                            grad0  = (2*(pred_mdl - y_data))/coef
                            compute_and_set_gradients(model,pred_mdl,dt,grad0, jacs = jacs)
                            optimizer.step()

                    # statistics
                    epoch_samples += x_data.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples
                if phase == 'train':
                    loss_train.append(epoch_loss)
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("LR", param_group['lr'])
                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    loss_val.append(epoch_loss)
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
    return best_model_wts_ret,model, loss_train, loss_val