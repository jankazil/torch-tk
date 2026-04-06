'''
Loss-related utilities for evaluating models at per-sample resolution.

This module provides functions to compute per-sample losses from either
a data loader or in-memory tensors, preserve the model's training state
during evaluation, and identify the samples with the largest losses.
'''

import torch

from torch_tk.models.utils import get_model_device


def per_sample_loss_from_data_loader(model, loss_function_sample_resolved, data_loader):
    '''
    Compute per-sample loss values and their mean from a data loader.

    The model is evaluated without gradient tracking, its original training
    state is restored afterward, and the returned per-sample loss tensor is
    moved to CPU memory.

    Returns
    -------
    float
        Mean loss over all samples in the data loader.
    torch.Tensor
        A 1-D tensor containing the model loss for each sample in the given data loader,
        always in CPU memory.
    '''

    device = get_model_device(model)

    was_training = model.training

    model.eval()

    data_n = len(data_loader.dataset)

    per_sample_loss = torch.empty(data_n, device=device)

    with torch.no_grad():
        ii = 0

        for x, y in data_loader:
            batch_n = len(x)
            loss = loss_function_sample_resolved(model(x.to(device)), y.to(device))
            if loss.ndim != 1:
                raise ValueError('Per-sample loss function does not produce a 1-dimensional tensor.')
            if loss.shape[0] != batch_n:
                raise ValueError('Per-sample loss function produces fewer or more loss values than samples.')
            per_sample_loss[ii : ii + batch_n] = loss
            ii += batch_n

        mean_per_sample_loss = per_sample_loss.mean().item()

    model.train(was_training)

    return mean_per_sample_loss, per_sample_loss.detach().cpu()


def per_sample_loss_from_data(model, loss_function_sample_resolved, x_data, y_data, chunk_size=None):
    '''
    Compute per-sample loss values and their mean from in-memory tensors.

    The model is evaluated without gradient tracking, optionally in chunks to
    limit memory usage, and its original training state is restored afterward.
    The returned per-sample loss tensor is moved to CPU memory.

    Arguments
    ----------
    model : torch.nn.Module
        Model used for prediction.
    loss_function_sample_resolved : callable
        Function taking (predictions, targets) and returning a 1-D tensor
        of per-sample losses.
    x_data : torch.Tensor
        Input data of shape (N, ...).
    y_data : torch.Tensor
        Target data of shape (N, ...).
    chunk_size : int, optional
        Number of samples to process at once. If not provided, all samples will be processed at once.

    Returns
    -------
    float
        Mean loss over all samples.
    torch.Tensor
        A 1-D tensor containing the loss for each sample, always in CPU memory.
    '''

    device = get_model_device(model)

    was_training = model.training

    model.eval()

    data_n = x_data.shape[0]

    per_sample_loss = torch.empty(data_n, device=device)

    with torch.no_grad():
        if chunk_size:
            for i_start in range(0, data_n, chunk_size):
                i_end = min(i_start + chunk_size, data_n)
                xb = x_data[i_start:i_end].to(device)
                yb = y_data[i_start:i_end].to(device)
                per_sample_loss[i_start:i_end] = loss_function_sample_resolved(model(xb), yb)
        else:
            per_sample_loss = loss_function_sample_resolved(model(x_data.to(device)), y_data.to(device))

    mean_per_sample_loss = per_sample_loss.mean().item()

    if per_sample_loss.ndim != 1 or per_sample_loss.shape[0] != data_n:
        raise ValueError('loss_function_sample_resolved must return a 1-D tensor with one loss value per sample.')

    model.train(was_training)

    return mean_per_sample_loss, per_sample_loss.detach().cpu()


def model_worst_loss(model, loss_function_sample_resolved, x_data, y_data, n, chunk_size=None):
    '''
    Return the indices and loss values of the n worst-performing samples.

    Calculates the
    - the n indices in the inputs x_data for which the given model has the
      largest loss relative to the reference data y_data
    - the corresponding loss values

    Arguments
    ----------
    model : torch.nn.Module
        The model to evaluate.
    x_data : torch.Tensor
        Input data, with batch dimension first.
    y_data : torch.Tensor
        Target data, with batch dimension first.
    chunk_size : int, optional
        Number of samples to process at once.

    Returns
    -------
    - the n indices in the inputs x_data for which the given model has the
      largest mean square error relative to the reference data y_data
    - the corresponding mean square errors
    '''

    # Model mean squared error for each data sample
    with torch.no_grad():
        mean_per_sample_loss, per_sample_loss = per_sample_loss_from_data(
            model, loss_function_sample_resolved, x_data, y_data, chunk_size=chunk_size
        )

    # Indices such that per_sample_loss[idxs] is sorted in descending order
    idxs = torch.argsort(per_sample_loss, descending=True)

    # The indices of the n elements in x_data that produce the largest model mean square error
    idxs_worst = idxs[:n].tolist()

    return idxs_worst, per_sample_loss[idxs_worst]
