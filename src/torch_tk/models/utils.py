'''
Utility functions for working with model objects.
'''


def get_model_device(model):
    '''
    Return the device on which the model is stored.

    The device is inferred from the first parameter or buffer found on the
    model. Raises an error if the model has neither parameters nor buffers.
    '''

    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    raise ValueError('Model has no parameters or buffers, so device is undefined.')
