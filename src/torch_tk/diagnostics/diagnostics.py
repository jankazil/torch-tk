'''
Utilities for storing, combining, and serializing sample-resolved training
diagnostics.

This module defines the Diagnostics class, which can be created from in-memory
data, a data loader, or a saved netCDF file. It stores per-sample loss values
together with model and optimizer metadata and can write the diagnostics back
to netCDF.
'''

from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .loss import per_sample_loss_from_data, per_sample_loss_from_data_loader


class Diagnostics:
    '''
    Store sample-resolved loss diagnostics together with training metadata.

    An instance holds model and optimizer identifiers, learning rate, batch
    size, an optional description, one or more epoch values, and the
    corresponding per-sample loss tensors. It also supports loading from and
    saving to netCDF.
    '''

    @classmethod
    def from_data_loader(
        cls,
        model,
        loss_function_sample_resolved,
        optimizer,
        learning_rate,
        batch_size,
        data_loader,
        description: str = None,
        epoch=0,
    ):
        '''
        Create a Diagnostics instance from a model evaluated on a data loader.

        The method computes per-sample losses for all samples in the loader and
        stores them together with model, optimizer, and training metadata.
        '''

        mean_per_sample_loss, per_sample_loss = per_sample_loss_from_data_loader(
            model, loss_function_sample_resolved, data_loader
        )

        return cls(
            type(model).__name__,
            type(optimizer).__name__,
            learning_rate,
            batch_size,
            epoch=epoch,
            per_sample_loss=per_sample_loss,
            description=description,
        )

    @classmethod
    def from_data(
        cls,
        model,
        loss_function_sample_resolved,
        optimizer,
        learning_rate,
        batch_size,
        x_data,
        y_data,
        description: str = None,
        epoch=0,
        chunk_size=None,
    ):
        '''
        Create a Diagnostics instance from in-memory input and target tensors.

        The method computes per-sample losses from the supplied tensors,
        optionally in chunks to reduce memory usage, and stores them
        together with model, optimizer, and training metadata.
        '''

        mean_per_sample_loss, per_sample_loss = per_sample_loss_from_data(
            model, loss_function_sample_resolved, x_data, y_data, chunk_size=chunk_size
        )

        return cls(
            type(model).__name__,
            type(optimizer).__name__,
            learning_rate,
            batch_size,
            epoch=epoch,
            per_sample_loss=per_sample_loss,
            description=description,
        )

    @classmethod
    def from_netcdf(cls, path):
        '''
        Create a Diagnostics instance from a saved netCDF file.

        This restores metadata, epoch values, and per-sample loss data from a
        previously saved diagnostics file, which is useful when resuming work
        from a checkpoint.
        '''
        if not isinstance(path, Path):
            path = Path(path)

        ds = xr.open_dataset(path)

        instance = cls(
            ds.attrs['model'],
            ds.attrs['optimizer'],
            ds.attrs['learning_rate'],
            ds.attrs['batch_size'],
            description=ds.attrs['description'],
            per_sample_loss=torch.as_tensor(ds['per_sample_loss'].values),
            epoch=ds['epoch'].values,
        )

        ds.close()

        return instance

    def __init__(
        self, model_name, optimizer_name, learning_rate, batch_size, epoch=None, per_sample_loss=None, description: str = None
    ):
        '''
        Initialize a Diagnostics instance.

        Epoch values and per-sample losses must either both be provided or both
        be omitted. If per-sample loss is one-dimensional, it is promoted to
        two dimensions so that the leading dimension corresponds to epoch.
        '''

        if epoch is None and per_sample_loss is not None:
            raise ValueError('epoch and per_sample_loss must both be None or both not None.')
        if epoch is not None and per_sample_loss is None:
            raise ValueError('epoch and per_sample_loss must both be None or both not None.')

        self.model = model_name
        self.optimizer = optimizer_name

        self.learning_rate = str(learning_rate)
        self.batch_size = batch_size
        self.description = description

        if epoch is not None:
            self.epoch = np.atleast_1d(np.asarray(epoch, dtype=np.int64))
        else:
            self.epoch = None

        # Add a first dimension that corresponds to the value(s) in the list self.epoch
        if per_sample_loss is not None:
            if per_sample_loss.ndim == 1:
                self.per_sample_loss = per_sample_loss.unsqueeze(0)
            else:
                self.per_sample_loss = per_sample_loss
        else:
            self.per_sample_loss = None

        # Check for consistency
        if epoch is not None and len(self.epoch) != self.per_sample_loss.shape[0]:
            raise ValueError('Number of epochs and number of per-sample loss instances do not match.')

        return

    def __add__(self, other):
        if not isinstance(other, Diagnostics):
            return NotImplemented

        '''
        Combine two compatible diagnostics objects.

        The two objects must have matching metadata. When both contain per-sample
        loss data, their epoch arrays and per-sample loss tensors are concatenated
        along the epoch dimension.
        '''

        result = Diagnostics.__new__(Diagnostics)

        assert self.model == other.model, (
            f"Cannot add Diagnostics objects with different models: "
            f"{self.model!r} != {other.model!r}"
        )
        assert self.optimizer == other.optimizer, (
            f"Cannot add Diagnostics objects with different optimizers: "
            f"{self.optimizer!r} != {other.optimizer!r}"
        )
        assert self.learning_rate == other.learning_rate, (
            f"Cannot add Diagnostics objects with different learning rates: "
            f"{self.learning_rate!r} != {other.learning_rate!r}"
        )
        assert self.batch_size == other.batch_size, (
            f"Cannot add Diagnostics objects with different batch sizes: "
            f"{self.batch_size!r} != {other.batch_size!r}"
        )
        assert self.description == other.description, (
            f"Cannot add Diagnostics objects with different descriptions: "
            f"{self.description!r} != {other.description!r}"
        )

        if self.per_sample_loss is None:
            result.model = other.model
            result.optimizer = other.optimizer
            result.epoch = other.epoch
            result.learning_rate = other.learning_rate
            result.batch_size = other.batch_size
            result.description = other.description
            result.per_sample_loss = other.per_sample_loss
        elif other.per_sample_loss is None:
            result.model = self.model
            result.optimizer = self.optimizer
            result.epoch = self.epoch
            result.learning_rate = self.learning_rate
            result.batch_size = self.batch_size
            result.description = self.description
            result.per_sample_loss = self.per_sample_loss
        else:
            result.model = self.model
            result.optimizer = self.optimizer
            result.epoch = np.concatenate((self.epoch, other.epoch))
            result.learning_rate = self.learning_rate
            result.batch_size = self.batch_size
            result.description = self.description
            result.per_sample_loss = torch.cat((self.per_sample_loss, other.per_sample_loss), dim=0)

        return result

    def to_netcdf(self, directory, verbose=True):
        '''
        Save the diagnostics object to a netCDF file.

        The output file is written under the given directory, which is created
        if necessary. The saved dataset includes epoch and sample coordinates,
        metadata attributes, and the per-sample loss array.
        '''

        if not isinstance(directory, Path):
            directory = Path(directory)

        file_name = Path(
            self.model
            + '.'
            + self.optimizer
            + '.'
            + self.description
            + '.epoch='
            + str(self.epoch[0])
            + '_to_'
            + str(self.epoch[-1])
            + '.nc'
        )

        # Create Xarray dataset
        ds = xr.Dataset()

        # Coordinates
        ds.coords['epoch'] = self.epoch
        ds.coords['sample'] = np.arange(self.per_sample_loss.shape[1])

        # Global attributes
        ds.attrs['model'] = self.model
        ds.attrs['optimizer'] = self.optimizer
        ds.attrs['description'] = self.description
        ds.attrs['learning_rate'] = self.learning_rate
        ds.attrs['batch_size'] = self.batch_size

        # 2D variables
        ds['per_sample_loss'] = (['epoch', 'sample'], self.per_sample_loss.detach().cpu().numpy())
        ds['per_sample_loss'].attrs['long_name'] = 'Mean per-sample loss'

        # Save
        file_path = directory / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(file_path, unlimited_dims='epoch')

        # Close
        ds.close()

        if verbose:
            print('Saved diagnostics in ', file_path)

        return file_path
