'''
Utility functions for working with PyTorch objects.
'''

import torch


def get_model_device(model: torch.nn.Module):
    '''
    Return the device of the first model parameter or, if none, the first buffer.

    Args:
        model: PyTorch module whose device is to be determined. Parameters and
            buffers in submodules are included. If the model spans multiple
            devices, only the first device found is returned.

    Raises:
        ValueError: If the model has neither parameters nor buffers.
    '''

    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    raise ValueError('Model has no parameters or buffers, so device is undefined.')
