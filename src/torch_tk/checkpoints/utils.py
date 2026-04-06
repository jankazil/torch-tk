import importlib


def class_path_of_instance(instance):
    '''
    Return the fully qualified class path for an instance's class.
    '''
    cls = type(instance)
    return cls.__module__ + '.' + cls.__qualname__


def import_class(class_path):
    '''
    Import and return a class from a fully qualified class path.

    Nested classes referenced through __qualname__ are supported.
    '''
    parts = class_path.split('.')

    for i in range(len(parts), 0, -1):
        module_name = '.'.join(parts[:i])
        try:
            obj = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            continue
    else:
        raise ImportError(f'Could not import any module prefix from {class_path!r}')

    for attr in parts[i:]:
        obj = getattr(obj, attr)

    return obj
