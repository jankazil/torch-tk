# Torch ToolKit (torch-tk)

**torch-tk** streamlines training, checkpoint management, and diagnostics of PyTorch models.

## Overview

The main idea behind **torch-tk** is that it adds a small amount of structure around ordinary torch models and optimizers so that they become easier to save, restore, reconstruct, and train. In particular, **torch-tk** provides a model base class, a checkpoint manager, a trainer, and diagnostics utilities.

**torch-tk** models can describe themselves. A model derived from `torch_tk.models.Model` provides the information needed to save its state and recreate the same model instance later. In practice, this means the model can expose its trainable parameters, return the constructor arguments needed to rebuild it, and load a saved state back into a fresh instance.

A second useful feature is the `CheckPointManager`. Rather than saving only raw parameter tensors, it saves enough information to reconstruct both the model and the optimizer in the state that created the checkpoint. This makes it easier to stop and resume training without manually rebuilding all objects.

**torch-tk** also includes a `Trainer` class for running epoch-based training. It supports training either from a `DataLoader` or directly from tensors, and it records simple diagnostics such as training loss and epoch wallclock time.

In addition, **torch-tk** provides a `Diagnostics` class for storing sample-resolved loss information together with training metadata. These diagnostics can be created from tensors or data loaders and can be saved to netCDF for later analysis.

## Classes

- `Model`
  
  - A base class which makes models self-describing and automatically reconstructible by the `CheckPointManager`
  - Automatically rebuilds a model from a saved file

- `SGD`, `Adam`, ...

   Wrapper classes for PyTorch optimizers that make the optimizers self-describing and automatically reconstructible by the `CheckPointManager`

- `Trainer`
  
  - Trains a model from
    - a `torch.utils.data.DataLoader`
    - or directly from tensors, using an efficient batching mechanism
  - Records training loss and model timing per epoch

- `CheckPointManager`
  
  - Saves and restores model training states
  - Automatically rebuilds both a model and its optimizer from a saved checkpoint file

- `Diagnostics`

  - Computes, stores, and plots per-sample loss and per-sample loss probability distribution
  - Saves and restores diagnostics in netCDF file format
  - Identifies worst-loss samples

## Installation

```bash
mamba install -c jan.kazil -c conda-forge torch-tk
```

## Public API

### Modules

#### `torch_tk.models.model`

Provides the abstract `Model` base class for models that can describe, save, restore, and reconstruct themselves.

- `Model.forward(xb)`: Abstract forward pass.
- `Model.named_parameters()`: Abstract iterator over `(name, parameter)` pairs.
- `Model.constructor_dict()`: Abstract description of constructor arguments needed for reconstruction.
- `Model.train(mode=True)` / `Model.eval()`: Training-mode control.
- `Model.parameters()`: Return trainable parameters as a list.
- `Model.state_dict(keep_vars=False)`: Return the current model state.
- `Model.load_state_dict(state_dict, strict=True)`: Load parameters from a state dictionary.
- `Model.save_state_dict_to_file(path)`: Save only the state dictionary.
- `Model.save_to_file(path)`: Save constructor arguments and state dictionary needed to recreate the model.
- `Model.load_from_file(path, device=None)`: Recreate a model from a saved file.
- `Model.clone(constructor_dict, state_dict, device=None)`: Reconstruct a model from constructor arguments and state.
- `Model.to(device)`: Move named parameters to a device.

#### `torch_tk.optimizers.sgd`

Provides a reconstruction-capable wrapper around `torch.optim.SGD`.

- `SGD(...)`: Subclass of `torch.optim.SGD` that stores its constructor arguments on the instance.
- `SGD.constructor_dict()`: Return the stored optimizer constructor settings excluding `params`.

#### `torch_tk.optimizers.adam`

Provides a reconstruction-capable wrapper around `torch.optim.Adam`.

- `Adam(...)`: Subclass of `torch.optim.Adam` that stores its constructor arguments on the instance.
- `Adam.constructor_dict()`: Return the stored optimizer constructor settings excluding `params`.

#### `torch_tk.training.trainer`

Provides the `Trainer` class for epoch-based training and simple training diagnostics.

- `Trainer(model, optimizer, loss_function, epoch=0)`: Initialize trainer state.
- `Trainer.train_with_dataloader(data_loader, num_epochs, epoch_diag_step=1, verbose=True)`: Train from a `DataLoader`.
- `Trainer.train_with_data(x_train, y_train, bs, num_epochs, epoch_diag_step=1, verbose=True, shuffle=False)`: Train from in-memory tensors.
- `Trainer.plot_loss(...)`: Plot recorded diagnostic loss versus epoch.
- `Trainer.plot_wallclock_time(...)`: Plot recorded epoch wallclock time versus epoch.

#### `torch_tk.checkpoints.checkpoint_manager`

Provides checkpoint management for saving and reconstructing a model and optimizer together.

- `CheckPointManager(model, optimizer, directory)`: Manage checkpoint saving in a directory.
- `CheckPointManager.save(epoch)`: Save a checkpoint containing epoch, class paths, constructor dictionaries, and state dictionaries.
- `CheckPointManager.load_from_file(file_path, device=None)`: Reconstruct and return `checkpoint_manager, model, optimizer, epoch` from a checkpoint file.

#### `torch_tk.diagnostics.loss`

Provides utilities for computing per-sample loss.

- `per_sample_loss_from_data_loader(model, loss_function_sample_resolved, data_loader)`: Compute per-sample losses and their mean from a `DataLoader`.
- `per_sample_loss_from_data(model, loss_function_sample_resolved, x_data, y_data, chunk_size=None)`: Compute per-sample losses and their mean from in-memory tensors.
- `model_worst_loss(model, loss_function_sample_resolved, x_data, y_data, n, chunk_size=None)`: Return the indices and values of the `n` worst losses.

#### `torch_tk.diagnostics.diagnostics`

Provides the `Diagnostics` container for sample-resolved loss diagnostics and analysis.

- `Diagnostics.from_data_loader(...)`: Build diagnostics from a model evaluated on a `DataLoader`.
- `Diagnostics.from_data(...)`: Build diagnostics from in-memory tensors.
- `Diagnostics.from_netcdf(path)`: Restore diagnostics from a saved netCDF file.
- `Diagnostics(...)`: Construct a diagnostics object from metadata, epochs, and per-sample loss data.
- `Diagnostics.__add__(other)`: Concatenate compatible diagnostics across epochs.
- `Diagnostics.to_netcdf(directory, verbose=True)`: Save diagnostics to a netCDF file.

#### `torch_tk.diagnostics.plotting`

Provides utilities for plotting diagnostics.

- `plot_diagnostics(diagnostics, plot_file=None, title=None, font_factor=1.5, figsize=(9, 6), xlim=None, ylim=None, loss_name='sqrt(loss)', pdf_bin_n=100, dpdlog10=False, show_plot=True, verbose=True)`: Plot kernel-density estimates of square-root per-sample loss distributions across one or more diagnostics objects and epochs.

## Notes and limitations

- The checkpoint mechanism assumes that models and optimizers are importable from stable class paths and expose `constructor_dict()`, `state_dict()`, and `load_state_dict()`.
- The checkpoint design is not suitable for optimizers that require non-serializable constructor inputs or custom parameter-group reconstruction beyond `model.parameters()`.
- The diagnostic plotting utility requires strictly positive, non-negative loss values because it plots the square root of loss on a logarithmic axis.
- The recorded epoch loss in `Trainer` is exact only when the supplied loss function returns the mean per-sample loss over each batch, as stated in the trainer docstrings.

## Development

### Code Quality and Testing Commands

- `make fmt` - Runs `ruff format`, which reformats Python files according to the style rules in `pyproject.toml`.
- `make lint` - Runs `ruff check --fix`, which lints the code and auto-fixes what it can.
- `make check` - Runs formatting and linting.
- `make type` - Currently disabled. Intended to run `mypy` using the settings in `pyproject.toml`.
- `make test` - Runs `pytest` with the test settings configured in `pyproject.toml`.

## Author

Jan Kazil - jan.kazil.dev@gmail.com - [jankazil.com](https://jankazil.com)

## License

BSD-3-Clause
