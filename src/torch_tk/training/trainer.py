'''
Utilities for training PyTorch models and recording simple diagnostics.

This module provides two epoch-based PyTorch training utilities:

- `Trainer`, for standard training when all target values are valid and the
  loss function returns a scalar loss.
- `MaskedTrainer`, for training when target tensors may contain missing values
  represented by NaN values. `MaskedTrainer` requires a masked loss function
  that excludes invalid target values and returns both the loss and the number
  of valid prediction-target pairs used to compute it.

Both trainers can run training either from a DataLoader or directly from
tensors. They store diagnostic losses and epoch wallclock times, and include
plotting utilities for recorded diagnostics.
'''

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_tk.models.utils import get_model_device


class Trainer:
    '''
    Utility for epoch-based training of a PyTorch model.

    The trainer operates with a given model, optimizer, and loss function,
    and supports mini-batch training either from a DataLoader or directly
    from input and target tensors. It tracks the current epoch and records
    simple per-diagnostic-epoch metrics, including training loss and epoch
    wallclock time.
    '''

    def __init__(self, model, optimizer, loss_function, epoch=0):
        '''
        Initialize the trainer state.

        Arguments
        ----------
        model : object
            Model to train.
        optimizer : object
            Optimizer used to update model parameters.
        loss_function : callable
            Function that computes the loss from predictions and targets.
        epoch : int, default=0
            Number of completed epochs at initialization.
        '''

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epoch = epoch  # Number of completed epochs

        self.diag_epochs = []  # Epoch numbers at which diagnostics were performed
        self.diag_epoch_train_losses = []  # Training data loss at epochs at which diagnostics were performed
        self.diag_epoch_valid_losses = []  # Validation data loss at epochs at which diagnostics were performed
        self.diag_epoch_wallclock_times = []  # Wallclock times for diagnostic epochs

    def train_with_dataloader(self, data_loader, num_epochs, epoch_diag_step=1, valid_data_loader=None, verbose=True):
        '''
        Train the model for a given number of epochs using a DataLoader.

        Batches yielded by the data loader do not need to already be on the
        same device as the model; each batch is moved to the model device
        before the forward pass.

        Diagnostics are computed and stored every `epoch_diag_step` epochs.
        The recorded epoch loss is recomputed over the full training dataset
        in evaluation mode. This loss is exact only when `loss_function`
        returns the mean per-sample loss over each batch.

        If a validation data loader is provided, additional diagnostics on the
        validation data are computed and stored every `epoch_diag_step` epochs.

        Arguments
        ----------
        data_loader : DataLoader
            Iterable yielding `(x_data, y_data)` batches for training.
        num_epochs : int
            Number of epochs to run.
        epoch_diag_step : int, default=1
            Frequency, in epochs, at which diagnostics are computed and stored.
        verbose : bool, default=True
            If True, print diagnostic information during training.
        valid_data_loader : DataLoader, default=None
            Iterable yielding `(x_data, y_data)` batches for validation.
        '''

        device = get_model_device(self.model)

        start_epoch = self.epoch

        for _ in range(num_epochs):
            self.epoch += 1

            if (self.epoch - start_epoch) % epoch_diag_step == 0:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

            self.model.train()

            for x_data, y_data in data_loader:
                x_data = x_data.to(device)
                y_data = y_data.to(device)

                self.optimizer.zero_grad()

                loss = self.loss_function(self.model(x_data), y_data)

                loss.backward()
                self.optimizer.step()

            if (self.epoch - start_epoch) % epoch_diag_step == 0:
                # Wallclock time for the most recent epoch
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                epoch_wallclock_time = t1 - t0

                self.diag_epochs.append(self.epoch)
                self.diag_epoch_wallclock_times.append(epoch_wallclock_time)

                # Calculate the loss over the entire training dataset. This is
                # exact only when the loss function returns the mean of a per-sample
                # loss over the batch, so that multiplying by batch size recovers
                # the batch sum of sample losses.

                self.model.eval()

                epoch_loss = 0.0

                with torch.no_grad():
                    for x_data, y_data in data_loader:
                        x_data = x_data.to(device)
                        y_data = y_data.to(device)

                        loss = self.loss_function(self.model(x_data), y_data)
                        epoch_loss += loss.item() * x_data.shape[0]

                epoch_loss /= len(data_loader.dataset)

                self.diag_epoch_train_losses.append(epoch_loss)

                # Calculate the loss over the entire validation dataset. This is
                # exact only when the loss function returns the mean of a per-sample
                # loss over the batch, so that multiplying by batch size recovers
                # the batch sum of sample losses.

                if valid_data_loader is not None:
                    self.model.eval()

                    epoch_valid_loss = 0.0

                    with torch.no_grad():
                        for x_data, y_data in valid_data_loader:
                            x_data = x_data.to(device)
                            y_data = y_data.to(device)

                            loss = self.loss_function(self.model(x_data), y_data)
                            epoch_valid_loss += loss.item() * x_data.shape[0]

                    epoch_valid_loss /= len(valid_data_loader.dataset)

                    self.diag_epoch_valid_losses.append(epoch_valid_loss)

                if verbose:
                    if valid_data_loader is None:
                        print(
                            f'\rTraining epoch {self.epoch} wallclock time (s): {epoch_wallclock_time:.4E}, training loss : {epoch_loss:.6f}',
                            end='\n',
                        )
                    else:
                        print(
                            f'\rTraining epoch {self.epoch} wallclock time (s): {epoch_wallclock_time:.4E}, training loss : {epoch_loss:.6f}, validation loss : {epoch_valid_loss:.6f}',
                            end='\n',
                        )

        return

    def train_with_data(
        self, x_train, y_train, bs, num_epochs, epoch_diag_step=1, x_valid=None, y_valid=None, shuffle=True, verbose=True
    ):
        '''
        Train the model for a given number of epochs using input and target tensors.

        The input and target tensors do not need to already be on the same
        device as the model; each selected batch is moved to the model device
        before the forward pass.

        By default, the training data are shuffled at the start of each epoch
        (shuffle = True). Diagnostics are computed and stored every `epoch_diag_step`
        epochs. The recorded epoch loss is recomputed over the full training
        dataset in evaluation mode and is exact only when `loss_function`
        returns the mean per-sample loss over each batch.

        If both x_valid and y_valid are provided, additional diagnostics on the
        validation data are computed and stored every `epoch_diag_step` epochs.

        Arguments
        ----------
        x_train : torch.Tensor
            Training inputs.
        y_train : torch.Tensor
            Training targets corresponding to `x_train`.
        bs : int
            Mini-batch size.
        num_epochs : int
            Number of epochs to run.
        epoch_diag_step : int, default=1
            Frequency, in epochs, at which diagnostics are computed and stored.
        verbose : bool, default=True
            If True, print diagnostic information during training.
        shuffle : bool, default=True
            If True, shuffle the training data at the start of each epoch.
        x_valid : torch.Tensor, default=None
            Validation input.
        y_valid : torch.Tensor, default=None
            Validation targets corresponding to `x_valid`.
        '''

        device = get_model_device(self.model)

        data_device = x_train.device

        train_data_n = x_train.shape[0]

        start_epoch = self.epoch

        for _ in range(num_epochs):
            self.epoch += 1

            if (self.epoch - start_epoch) % epoch_diag_step == 0:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

            self.model.train()

            # Create a random permutation of the indices selecting model input
            if shuffle:
                perm = torch.randperm(train_data_n, device=data_device)
            else:
                perm = torch.tensor(range(train_data_n), device=data_device)

            # Iterate over all full batches, plus one final (possibly smaller) batch
            for i in range((train_data_n - 1) // bs + 1):
                # Select batch
                start_i = i * bs
                end_i = (
                    start_i + bs
                )  # the last end_i may be greater than the number of training instances; slicing used next keeps indices in bounds

                indexes = perm[start_i:end_i]

                x_data = x_train[indexes].to(device)
                y_data = y_train[indexes].to(device)

                self.optimizer.zero_grad()

                loss = self.loss_function(self.model(x_data), y_data)

                loss.backward()
                self.optimizer.step()

            if (self.epoch - start_epoch) % epoch_diag_step == 0:
                # Wallclock time for the most recent epoch
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                epoch_wallclock_time = t1 - t0

                self.diag_epochs.append(self.epoch)
                self.diag_epoch_wallclock_times.append(epoch_wallclock_time)

                # Calculate the loss over the entire training dataset. This is
                # exact only when the loss function returns the mean of a per-sample
                # loss over the batch, so that multiplying by batch size recovers
                # the batch sum of sample losses.

                self.model.eval()

                epoch_loss = 0.0

                with torch.no_grad():
                    # Iterate over all full batches, plus one final (possibly smaller) batch
                    for i in range((train_data_n - 1) // bs + 1):
                        # Select batch
                        start_i = i * bs
                        end_i = (
                            start_i + bs
                        )  # the last end_i may be greater than the number of training instances; slicing used next keeps indices in bounds

                        indexes = perm[start_i:end_i]

                        x_data = x_train[indexes].to(device)
                        y_data = y_train[indexes].to(device)

                        loss = self.loss_function(self.model(x_data), y_data)

                        epoch_loss += loss.item() * x_data.shape[0]

                epoch_loss /= train_data_n

                self.diag_epoch_train_losses.append(epoch_loss)

                # Calculate the loss over the entire validation dataset. This is
                # exact only when the loss function returns the mean of a per-sample
                # loss over the batch, so that multiplying by batch size recovers
                # the batch sum of sample losses.

                if x_valid is not None and y_valid is not None:
                    self.model.eval()

                    valid_data_n = x_valid.shape[0]

                    epoch_valid_loss = 0.0

                    with torch.no_grad():
                        # Iterate over all full batches, plus one final (possibly smaller) batch
                        for i in range((valid_data_n - 1) // bs + 1):
                            # Select batch
                            start_i = i * bs
                            end_i = (
                                start_i + bs
                            )  # the last end_i may be greater than the number of validation instances; slicing used next keeps indices in bounds

                            x_data = x_valid[start_i:end_i].to(device)
                            y_data = y_valid[start_i:end_i].to(device)

                            loss = self.loss_function(self.model(x_data), y_data)

                            epoch_valid_loss += loss.item() * x_data.shape[0]

                    epoch_valid_loss /= valid_data_n

                    self.diag_epoch_valid_losses.append(epoch_valid_loss)

                if verbose:
                    if x_valid is None or y_valid is None:
                        print(
                            f'\rTraining epoch {self.epoch} wallclock time (s): {epoch_wallclock_time:.4E}, training loss : {epoch_loss:.6f}',
                            end='\n',
                        )
                    else:
                        print(
                            f'\rTraining epoch {self.epoch} wallclock time (s): {epoch_wallclock_time:.4E}, training loss : {epoch_loss:.6f}, validation loss : {epoch_valid_loss:.6f}',
                            end='\n',
                        )

        return

    def plot_loss(
        self,
        plot_file=None,
        title=None,
        font_factor=1.5,
        figsize=(9, 6),
        xlim=None,
        ylim=None,
        xlog=False,
        ylog=False,
        xlabel='Epoch',
        ylabel='Loss',
        show_plot=True,
        verbose=True,
    ) -> Path:
        '''
        Plot the recorded training loss versus epoch.

        Arguments
        ----------
        plot_file : str or path-like, optional
            Output path for saving the figure. If omitted, the plot is not saved.
        title : str, optional
            Figure title.
        font_factor : float, default=1.5
            Multiplicative factor applied to all font sizes in the figure.
        figsize : tuple, default=(9, 6)
            Figure size passed to Matplotlib.
        xlim : tuple, optional
            Limits for the x-axis.
        ylim : tuple, optional
            Limits for the y-axis.
        xlog:
            If True, use a logarithmic x-axis.
        ylog:
            If True, use a logarithmic y-axis.
        xlabel : str, default='Epoch'
            Label for the x-axis.
        ylabel : str, default='Loss'
            Label for the y-axis.
        show_plot : bool, default=True
            If True, display the figure; otherwise close it after saving.
        verbose : bool, default=True
            If True, print the output path of the created plot.
        '''

        fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, squeeze=False)

        plt.suptitle(title, y=0.94, fontsize=8)

        ax[0, 0].plot(self.diag_epochs, self.diag_epoch_train_losses, color='r', marker='o', linestyle='None', label='Training')

        if len(self.diag_epoch_valid_losses) > 0:
            ax[0, 0].plot(
                self.diag_epochs, self.diag_epoch_valid_losses, color='blue', marker='o', linestyle='None', label='Validation'
            )

        if xlog:
            ax[0, 0].set_xscale('log')
        else:
            ax[0, 0].set_xscale('linear')

        if ylog:
            ax[0, 0].set_yscale('log')
        else:
            ax[0, 0].set_yscale('linear')

        if xlim:
            ax[0, 0].set_xlim(xlim)
        else:
            ax[0, 0].set_xlim([min(self.diag_epochs), max(self.diag_epochs)])
        if ylim:
            ax[0, 0].set_ylim(ylim)
        else:
            if len(self.diag_epoch_valid_losses) == 0:
                ax[0, 0].set_ylim([min(self.diag_epoch_train_losses), max(self.diag_epoch_train_losses)])
            else:
                ax[0, 0].set_ylim(
                    [
                        min(self.diag_epoch_train_losses + self.diag_epoch_valid_losses),
                        max(self.diag_epoch_train_losses + self.diag_epoch_valid_losses),
                    ]
                )

        ax[0, 0].legend(loc='best', fontsize=7, frameon=False)

        ax[0, 0].set_xlabel(xlabel)
        ax[0, 0].set_ylabel(ylabel)

        # Increase all font sizes by a given factor
        for text in fig.findobj(match=lambda artist: hasattr(artist, 'get_fontsize')):
            text.set_fontsize(text.get_fontsize() * font_factor)

        if plot_file:
            plot_file = Path(plot_file)
            plot_dir = plot_file.parent
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(plot_file, bbox_inches='tight', pad_inches=0.25)
            if verbose:
                print('Created plot', plot_file)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return plot_file

    def plot_wallclock_time(
        self,
        plot_file=None,
        title=None,
        font_factor=1.5,
        figsize=(9, 6),
        xlim=None,
        ylim=None,
        xlog=False,
        ylog=False,
        xlabel='Epoch',
        ylabel='Wallclock time (s)',
        show_plot=True,
        verbose=True,
    ) -> Path:
        '''
        Plot the recorded epoch wallclock time versus epoch.

        Arguments
        ----------
        plot_file : str or path-like, optional
            Output path for saving the figure. If omitted, the plot is not saved.
        title : str, optional
            Figure title.
        font_factor : float, default=1.5
            Multiplicative factor applied to all font sizes in the figure.
        figsize : tuple, default=(9, 6)
            Figure size passed to Matplotlib.
        xlim : tuple, optional
            Limits for the x-axis.
        ylim : tuple, optional
            Limits for the y-axis.
        xlog:
            If True, use a logarithmic x-axis.
        ylog:
            If True, use a logarithmic y-axis.
        xlabel : str, default='Epoch'
            Label for the x-axis.
        ylabel : str, default='Wallclock time (s)'
            Label for the y-axis.
        show_plot : bool, default=True
            If True, display the figure; otherwise close it after saving.
        verbose : bool, default=True
            If True, print the output path of the created plot.
        '''

        fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, squeeze=False)

        plt.suptitle(title, y=0.94, fontsize=8)

        ax[0, 0].plot(self.diag_epochs, self.diag_epoch_wallclock_times, color='r', marker='o', linestyle='None')

        if xlog:
            ax[0, 0].set_xscale('log')
        else:
            ax[0, 0].set_xscale('linear')

        if ylog:
            ax[0, 0].set_yscale('log')
        else:
            ax[0, 0].set_yscale('linear')

        if xlim:
            ax[0, 0].set_xlim(xlim)
        else:
            ax[0, 0].set_xlim([min(self.diag_epochs), max(self.diag_epochs)])
        if ylim:
            ax[0, 0].set_ylim(ylim)
        else:
            ax[0, 0].set_ylim([min(self.diag_epoch_wallclock_times), max(self.diag_epoch_wallclock_times)])

        ax[0, 0].set_xlabel(xlabel)
        ax[0, 0].set_ylabel(ylabel)

        # Increase all font sizes by a given factor
        for text in fig.findobj(match=lambda artist: hasattr(artist, 'get_fontsize')):
            text.set_fontsize(text.get_fontsize() * font_factor)

        if plot_file:
            plot_file = Path(plot_file)
            plot_dir = plot_file.parent
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(plot_file, bbox_inches='tight', pad_inches=0.25)
            if verbose:
                print('Created plot', plot_file)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return plot_file


class MaskedTrainer:
    '''
    Utility for epoch-based training of a PyTorch model with missing-target
    support.

    The trainer operates with a given model, optimizer, and masked loss
    function. It supports mini-batch training either from a DataLoader or
    directly from input and target tensors. It tracks the current epoch and
    records per-diagnostic-epoch metrics, including training loss, optional
    validation loss, and epoch wallclock time.

    This class requires a `masked_loss_function(prediction, target)` callable
    that returns a two-element tuple: `(loss, target_valid_n)`. The loss must
    be the mean loss over valid prediction-target pairs only, where invalid
    target values are represented by NaN and excluded before subtraction,
    squaring, or any other elementwise loss operation is performed.
    `target_valid_n` must be the number of valid prediction-target pairs used
    to compute the returned loss. If a batch contains no valid target values,
    the callable must return `target_valid_n == 0`; its loss value may be NaN.

    Epoch-level training and validation losses are accumulated as weighted
    means using `target_valid_n`, so the loss function must return a mean over
    valid target elements rather than a sum.
    '''

    def __init__(self, model, optimizer, masked_loss_function, epoch=0):
        '''
        Initialize the trainer state.

        Arguments
        ----------
        model : object
            Model to train.
        optimizer : object
            Optimizer used to update model parameters.
        masked_loss_function : callable
            Callable with signature `masked_loss_function(prediction, target)`.
            It must return `(loss, target_valid_n)`, where `loss` is the mean
            loss over target elements that are not NaN and `target_valid_n` is
            the number of valid prediction-target pairs used in the loss. NaN
            target values must be masked out before subtraction, squaring, or
            any other elementwise loss operation. If a batch has no valid
            target values, `target_valid_n` must be 0 and the batch is skipped.
        epoch : int, default=0
            Number of completed epochs at initialization.
        '''

        self.model = model
        self.optimizer = optimizer
        self.masked_loss_function = masked_loss_function
        self.epoch = epoch  # Number of completed epochs

        self.diag_epochs = []  # Epoch numbers for which diagnostics were recorded
        self.diag_epoch_train_losses = []  # Training losses recorded at diagnostic epochs
        self.diag_epoch_valid_losses = []  # Validation losses recorded at diagnostic epochs
        self.diag_epoch_wallclock_times = []  # Wallclock durations of diagnostic epochs

    def train_with_dataloader(self, data_loader, num_epochs, epoch_diag_step=1, valid_data_loader=None, verbose=True):
        '''
        Train the model for a given number of epochs using a DataLoader.

        Batches yielded by the data loader do not need to already be on the
        same device as the model; each batch is moved to the model device
        before the forward pass.

        Diagnostics are computed and stored every `epoch_diag_step` epochs.
        The recorded epoch loss is recomputed over the full training dataset
        in evaluation mode. This loss is accumulated as a weighted mean over
        valid target elements, using the `target_valid_n` returned by
        `masked_loss_function`.

        If a validation data loader is provided, additional diagnostics on the
        validation data are computed and stored every `epoch_diag_step` epochs.

        Arguments
        ----------
        data_loader : DataLoader
            Iterable yielding `(x_data, y_data)` batches for training.
        num_epochs : int
            Number of epochs to run.
        epoch_diag_step : int, default=1
            Frequency, in epochs, at which diagnostics are computed and stored.
        verbose : bool, default=True
            If True, print diagnostic information during training.
        valid_data_loader : DataLoader, default=None
            Iterable yielding `(x_data, y_data)` batches for validation.
        '''

        device = get_model_device(self.model)

        start_epoch = self.epoch

        for _ in range(num_epochs):
            self.epoch += 1

            if (self.epoch - start_epoch) % epoch_diag_step == 0:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

            self.model.train()

            for x_data, y_data in data_loader:
                x_data = x_data.to(device)
                y_data = y_data.to(device)

                self.optimizer.zero_grad()

                loss, target_valid_n = self.masked_loss_function(self.model(x_data), y_data)

                # A batch with no valid target values contributes no usable loss.
                # Skip backpropagation and the parameter update for such batches.
                if target_valid_n == 0:
                    continue

                if torch.isnan(loss):
                    raise RuntimeError('Loss is NaN despite valid target values. Check model predictions.')

                loss.backward()
                self.optimizer.step()

            if (self.epoch - start_epoch) % epoch_diag_step == 0:
                # Wallclock time for the most recent epoch
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                epoch_wallclock_time = t1 - t0

                self.diag_epochs.append(self.epoch)
                self.diag_epoch_wallclock_times.append(epoch_wallclock_time)

                # Calculate the loss over the entire training dataset as a
                # weighted mean over valid target elements. This assumes the
                # masked loss returns a mean loss and the corresponding count
                # of valid prediction-target pairs.

                self.model.eval()

                epoch_loss = 0.0
                counter = 0

                with torch.no_grad():
                    for x_data, y_data in data_loader:
                        x_data = x_data.to(device)
                        y_data = y_data.to(device)

                        loss, target_valid_n = self.masked_loss_function(self.model(x_data), y_data)

                        if target_valid_n == 0:
                            continue

                        if torch.isnan(loss):
                            raise RuntimeError('Loss is NaN despite valid target values. Check model predictions.')

                        epoch_loss += loss.item() * target_valid_n
                        counter += target_valid_n

                if counter > 0:
                    epoch_loss /= counter
                else:
                    epoch_loss = np.nan

                self.diag_epoch_train_losses.append(epoch_loss)

                # Calculate the loss over the entire validation dataset as a
                # weighted mean over valid target elements. This assumes the
                # masked loss returns a mean loss and the corresponding count
                # of valid prediction-target pairs.

                if valid_data_loader is not None:
                    self.model.eval()

                    epoch_valid_loss = 0.0
                    counter = 0

                    with torch.no_grad():
                        for x_data, y_data in valid_data_loader:
                            x_data = x_data.to(device)
                            y_data = y_data.to(device)

                            loss, target_valid_n = self.masked_loss_function(self.model(x_data), y_data)

                            if target_valid_n == 0:
                                continue

                            if torch.isnan(loss):
                                raise RuntimeError('Loss is NaN despite valid target values. Check model predictions.')

                            epoch_valid_loss += loss.item() * target_valid_n
                            counter += target_valid_n

                    if counter > 0:
                        epoch_valid_loss /= counter
                    else:
                        epoch_valid_loss = np.nan

                    self.diag_epoch_valid_losses.append(epoch_valid_loss)

                if verbose:
                    if valid_data_loader is None:
                        print(
                            f'\rTraining epoch {self.epoch} wallclock time (s): {epoch_wallclock_time:.4E}, training loss : {epoch_loss:.6f}',
                            end='\n',
                        )
                    else:
                        print(
                            f'\rTraining epoch {self.epoch} wallclock time (s): {epoch_wallclock_time:.4E}, training loss : {epoch_loss:.6f}, validation loss : {epoch_valid_loss:.6f}',
                            end='\n',
                        )

        return

    def train_with_data(
        self, x_train, y_train, bs, num_epochs, epoch_diag_step=1, x_valid=None, y_valid=None, shuffle=True, verbose=True
    ):
        '''
        Train the model for a given number of epochs using input and target tensors.

        The input and target tensors do not need to already be on the same
        device as the model; each selected batch is moved to the model device
        before the forward pass.

        By default, the training data are shuffled at the start of each epoch
        (shuffle = True). Diagnostics are computed and stored every `epoch_diag_step`
        epochs. The recorded epoch loss is recomputed over the full training
        dataset in evaluation mode and is accumulated as a weighted mean over
        valid target elements, using the `target_valid_n` returned by
        `masked_loss_function`.

        If both x_valid and y_valid are provided, additional diagnostics on the
        validation data are computed and stored every `epoch_diag_step` epochs.

        Arguments
        ----------
        x_train : torch.Tensor
            Training inputs.
        y_train : torch.Tensor
            Training targets corresponding to `x_train`.
        bs : int
            Mini-batch size.
        num_epochs : int
            Number of epochs to run.
        epoch_diag_step : int, default=1
            Frequency, in epochs, at which diagnostics are computed and stored.
        verbose : bool, default=True
            If True, print diagnostic information during training.
        shuffle : bool, default=True
            If True, shuffle the training data at the start of each epoch.
        x_valid : torch.Tensor, default=None
            Validation inputs.
        y_valid : torch.Tensor, default=None
            Validation targets corresponding to `x_valid`.
        '''

        device = get_model_device(self.model)

        data_device = x_train.device

        train_data_n = x_train.shape[0]

        start_epoch = self.epoch

        for _ in range(num_epochs):
            self.epoch += 1

            if (self.epoch - start_epoch) % epoch_diag_step == 0:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

            self.model.train()

            # Create the index order used to select training mini-batches
            if shuffle:
                perm = torch.randperm(train_data_n, device=data_device)
            else:
                perm = torch.tensor(range(train_data_n), device=data_device)

            # Iterate over all full batches, plus one final (possibly smaller) batch
            for i in range((train_data_n - 1) // bs + 1):
                # Select the current training batch
                start_i = i * bs
                end_i = (
                    start_i + bs
                )  # the last end_i may be greater than the number of training instances; slicing used next keeps indices in bounds

                indexes = perm[start_i:end_i]

                x_data = x_train[indexes].to(device)
                y_data = y_train[indexes].to(device)

                self.optimizer.zero_grad()

                loss, target_valid_n = self.masked_loss_function(self.model(x_data), y_data)

                # A batch with no valid target values contributes no usable loss.
                # Skip backpropagation and the parameter update for such batches.
                if target_valid_n == 0:
                    continue

                if torch.isnan(loss):
                    raise RuntimeError('Loss is NaN despite valid target values. Check model predictions.')

                loss.backward()
                self.optimizer.step()

            if (self.epoch - start_epoch) % epoch_diag_step == 0:
                # Wallclock time for the most recent epoch
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                epoch_wallclock_time = t1 - t0

                self.diag_epochs.append(self.epoch)
                self.diag_epoch_wallclock_times.append(epoch_wallclock_time)

                # Calculate the loss over the entire training dataset as a
                # weighted mean over valid target elements. This assumes the
                # masked loss returns a mean loss and the corresponding count
                # of valid prediction-target pairs.

                self.model.eval()

                epoch_loss = 0.0
                counter = 0

                with torch.no_grad():
                    # Iterate over all full batches, plus one final (possibly smaller) batch
                    for i in range((train_data_n - 1) // bs + 1):
                        # Select the current training batch
                        start_i = i * bs
                        end_i = (
                            start_i + bs
                        )  # the last end_i may be greater than the number of training instances; slicing used next keeps indices in bounds

                        indexes = perm[start_i:end_i]

                        x_data = x_train[indexes].to(device)
                        y_data = y_train[indexes].to(device)

                        loss, target_valid_n = self.masked_loss_function(self.model(x_data), y_data)

                        if target_valid_n == 0:
                            continue

                        if torch.isnan(loss):
                            raise RuntimeError('Loss is NaN despite valid target values. Check model predictions.')

                        epoch_loss += loss.item() * target_valid_n
                        counter += target_valid_n

                if counter > 0:
                    epoch_loss /= counter
                else:
                    epoch_loss = np.nan

                self.diag_epoch_train_losses.append(epoch_loss)

                # Calculate the loss over the entire validation dataset as a
                # weighted mean over valid target elements. This assumes the
                # masked loss returns a mean loss and the corresponding count
                # of valid prediction-target pairs.

                if x_valid is not None and y_valid is not None:
                    self.model.eval()

                    valid_data_n = x_valid.shape[0]

                    epoch_valid_loss = 0.0
                    counter = 0

                    with torch.no_grad():
                        # Iterate over all full batches, plus one final (possibly smaller) batch
                        for i in range((valid_data_n - 1) // bs + 1):
                            # Select the current validation batch
                            start_i = i * bs
                            end_i = (
                                start_i + bs
                            )  # the last end_i may be greater than the number of validation instances; slicing used next keeps indices in bounds

                            x_data = x_valid[start_i:end_i].to(device)
                            y_data = y_valid[start_i:end_i].to(device)

                            loss, target_valid_n = self.masked_loss_function(self.model(x_data), y_data)

                            if target_valid_n == 0:
                                continue

                            if torch.isnan(loss):
                                raise RuntimeError('Loss is NaN despite valid target values. Check model predictions.')

                            epoch_valid_loss += loss.item() * target_valid_n
                            counter += target_valid_n

                    if counter > 0:
                        epoch_valid_loss /= counter
                    else:
                        epoch_valid_loss = np.nan

                    self.diag_epoch_valid_losses.append(epoch_valid_loss)

                if verbose:
                    if x_valid is None or y_valid is None:
                        print(
                            f'\rTraining epoch {self.epoch} wallclock time (s): {epoch_wallclock_time:.4E}, training loss : {epoch_loss:.6f}',
                            end='\n',
                        )
                    else:
                        print(
                            f'\rTraining epoch {self.epoch} wallclock time (s): {epoch_wallclock_time:.4E}, training loss : {epoch_loss:.6f}, validation loss : {epoch_valid_loss:.6f}',
                            end='\n',
                        )

        return

    def plot_loss(
        self,
        plot_file=None,
        title=None,
        font_factor=1.5,
        figsize=(9, 6),
        xlim=None,
        ylim=None,
        xlog=False,
        ylog=False,
        xlabel='Epoch',
        ylabel='Loss',
        show_plot=True,
        verbose=True,
    ) -> Path:
        '''
        Plot the recorded training and, if available, validation loss versus epoch.

        Arguments
        ----------
        plot_file : str or path-like, optional
            Output path for saving the figure. If omitted, the plot is not saved.
        title : str, optional
            Figure title.
        font_factor : float, default=1.5
            Multiplicative factor applied to all font sizes in the figure.
        figsize : tuple, default=(9, 6)
            Figure size passed to Matplotlib.
        xlim : tuple, optional
            Limits for the x-axis.
        ylim : tuple, optional
            Limits for the y-axis.
        xlog : bool, default=False
            If True, use a logarithmic x-axis.
        ylog : bool, default=False
            If True, use a logarithmic y-axis.
        xlabel : str, default='Epoch'
            Label for the x-axis.
        ylabel : str, default='Loss'
            Label for the y-axis.
        show_plot : bool, default=True
            If True, display the figure; otherwise close it after saving.
        verbose : bool, default=True
            If True, print the output path of the created plot.
        '''

        fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, squeeze=False)

        plt.suptitle(title, y=0.94, fontsize=8)

        ax[0, 0].plot(self.diag_epochs, self.diag_epoch_train_losses, color='r', marker='o', linestyle='None', label='Training')

        if len(self.diag_epoch_valid_losses) > 0:
            ax[0, 0].plot(
                self.diag_epochs, self.diag_epoch_valid_losses, color='blue', marker='o', linestyle='None', label='Validation'
            )

        if xlog:
            ax[0, 0].set_xscale('log')
        else:
            ax[0, 0].set_xscale('linear')

        if ylog:
            ax[0, 0].set_yscale('log')
        else:
            ax[0, 0].set_yscale('linear')

        if xlim:
            ax[0, 0].set_xlim(xlim)
        else:
            ax[0, 0].set_xlim([min(self.diag_epochs), max(self.diag_epochs)])
        if ylim:
            ax[0, 0].set_ylim(ylim)
        else:
            if len(self.diag_epoch_valid_losses) == 0:
                ax[0, 0].set_ylim([min(self.diag_epoch_train_losses), max(self.diag_epoch_train_losses)])
            else:
                ax[0, 0].set_ylim(
                    [
                        min(self.diag_epoch_train_losses + self.diag_epoch_valid_losses),
                        max(self.diag_epoch_train_losses + self.diag_epoch_valid_losses),
                    ]
                )

        ax[0, 0].legend(loc='best', fontsize=7, frameon=False)

        ax[0, 0].set_xlabel(xlabel)
        ax[0, 0].set_ylabel(ylabel)

        # Scale all detected figure text by the requested factor
        for text in fig.findobj(match=lambda artist: hasattr(artist, 'get_fontsize')):
            text.set_fontsize(text.get_fontsize() * font_factor)

        if plot_file:
            plot_file = Path(plot_file)
            plot_dir = plot_file.parent
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(plot_file, bbox_inches='tight', pad_inches=0.25)
            if verbose:
                print('Created plot', plot_file)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return plot_file

    def plot_wallclock_time(
        self,
        plot_file=None,
        title=None,
        font_factor=1.5,
        figsize=(9, 6),
        xlim=None,
        ylim=None,
        xlog=False,
        ylog=False,
        xlabel='Epoch',
        ylabel='Wallclock time (s)',
        show_plot=True,
        verbose=True,
    ) -> Path:
        '''
        Plot the recorded diagnostic-epoch wallclock time versus epoch.

        Arguments
        ----------
        plot_file : str or path-like, optional
            Output path for saving the figure. If omitted, the plot is not saved.
        title : str, optional
            Figure title.
        font_factor : float, default=1.5
            Multiplicative factor applied to all font sizes in the figure.
        figsize : tuple, default=(9, 6)
            Figure size passed to Matplotlib.
        xlim : tuple, optional
            Limits for the x-axis.
        ylim : tuple, optional
            Limits for the y-axis.
        xlog : bool, default=False
            If True, use a logarithmic x-axis.
        ylog : bool, default=False
            If True, use a logarithmic y-axis.
        xlabel : str, default='Epoch'
            Label for the x-axis.
        ylabel : str, default='Wallclock time (s)'
            Label for the y-axis.
        show_plot : bool, default=True
            If True, display the figure; otherwise close it after saving.
        verbose : bool, default=True
            If True, print the output path of the created plot.
        '''

        fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, squeeze=False)

        plt.suptitle(title, y=0.94, fontsize=8)

        ax[0, 0].plot(self.diag_epochs, self.diag_epoch_wallclock_times, color='r', marker='o', linestyle='None')

        if xlog:
            ax[0, 0].set_xscale('log')
        else:
            ax[0, 0].set_xscale('linear')

        if ylog:
            ax[0, 0].set_yscale('log')
        else:
            ax[0, 0].set_yscale('linear')

        if xlim:
            ax[0, 0].set_xlim(xlim)
        else:
            ax[0, 0].set_xlim([min(self.diag_epochs), max(self.diag_epochs)])
        if ylim:
            ax[0, 0].set_ylim(ylim)
        else:
            ax[0, 0].set_ylim([min(self.diag_epoch_wallclock_times), max(self.diag_epoch_wallclock_times)])

        ax[0, 0].set_xlabel(xlabel)
        ax[0, 0].set_ylabel(ylabel)

        # Scale all detected figure text by the requested factor
        for text in fig.findobj(match=lambda artist: hasattr(artist, 'get_fontsize')):
            text.set_fontsize(text.get_fontsize() * font_factor)

        if plot_file:
            plot_file = Path(plot_file)
            plot_dir = plot_file.parent
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(plot_file, bbox_inches='tight', pad_inches=0.25)
            if verbose:
                print('Created plot', plot_file)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return plot_file
