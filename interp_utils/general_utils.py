import torch
import sys
import traceback
import numpy as np
from typing import Union, List


def get_scheduler(optimizer, n_steps):
    def lr_lambda(step):
        if step < 0.05*n_steps:
            return step/(0.05*n_steps)
        else:
            return 1-(step-0.05*n_steps)/(n_steps-0.05*n_steps+1e-3)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def is_iterable(x):
    try:
        _ = iter(x)
        return True
    except:
        return False

def opt(x):
    # print elements of dir(x) that don't start with _
    items = [item for item in dir(x) if not item.startswith('_')]
    for item in items:
        print(item)


def reload_module_hack(module_name: Union[str, List[str]]):
    if isinstance(module_name, list):
        for m in module_name:
            reload_module_hack(m)
        return
    else:
        assert isinstance(module_name, str)
    for x in list(sys.modules):
        if x.split('.')[0] == module_name:
            del sys.modules[x]
    module = __import__(module_name, fromlist=[''])
    return module

def see(t):
    # renders array shape and name of argument
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    code = code.replace('see(', '')[:-1]
    print('>> '+code, end='')
    if isinstance(t, torch.Tensor):
        print(': '+str(tuple(t.shape)))
    elif isinstance(t, np.ndarray):
        print(': '+str(t.shape))
    elif isinstance(t, list) and (isinstance(t[0], torch.Tensor) or isinstance(t[0], np.ndarray)):
        print(': '+str([tuple(arr.shape) for arr in t]))
    else:
        print(t)


def asee(t: torch.Tensor):
    # short for array-see, renders array as well as shape
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    code = code.replace('see(', '')[:-1]
    print('> '+code, end='')
    if isinstance(t, torch.Tensor):
        print(': '+str(tuple(t.shape)))
        # print('line: '+str(lineno))
        print('arr: '+str(t.detach().numpy()))
    elif isinstance(t, np.ndarray):
        print(': '+str(t.shape))
        # print('line: '+str(lineno))
        print('arr: '+str(t))
    else:
        print(t)
        # print('line: '+str(lineno))
    print()