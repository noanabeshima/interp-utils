import torch
import sys
import traceback
import numpy as np
from typing import Union, List


def get_scheduler(optimizer, n_steps):
    def lr_lambda(step):
        if step < 0.05 * n_steps:
            return step / (0.05 * n_steps)
        else:
            return 1 - (step - 0.05 * n_steps) / (n_steps - 0.05 * n_steps + 1e-3)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def is_iterable(x):
    try:
        _ = iter(x)
        return True
    except:
        return False


def opt(x):
    # print elements of dir(x) that don't start with _
    items = [item for item in dir(x) if not item.startswith("_")]
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
        if x.split(".")[0] == module_name:
            del sys.modules[x]
    module = __import__(module_name, fromlist=[""])
    return module


def see(t):
    # renders array shape and name of argument
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    code = code.replace("see(", "")[:-1]
    print(">> " + code, end="")
    if isinstance(t, torch.Tensor):
        print(": " + str(tuple(t.shape)))
    elif isinstance(t, np.ndarray):
        print(": " + str(t.shape))
    elif isinstance(t, list) and (
        isinstance(t[0], torch.Tensor) or isinstance(t[0], np.ndarray)
    ):
        print(": " + str([tuple(arr.shape) for arr in t]))
    else:
        print(t)


def asee(t: torch.Tensor):
    # short for array-see, renders array as well as shape
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    code = code.replace("see(", "")[:-1]
    print("> " + code, end="")
    if isinstance(t, torch.Tensor):
        print(": " + str(tuple(t.shape)))
        # print('line: '+str(lineno))
        print("arr: " + str(t.detach().numpy()))
    elif isinstance(t, np.ndarray):
        print(": " + str(t.shape))
        # print('line: '+str(lineno))
        print("arr: " + str(t))
    else:
        print(t)
        # print('line: '+str(lineno))
    print()

def batched_bincount(x, dim, max_value):
    # From Guillaume Leclerc: https://discuss.pytorch.org/t/batched-bincount/72819/3
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


class TensorHistogramObserver:
    def __init__(self, min, max, bin_width, tensor_shape):
        '''
        Useful for storing bin counts for multiple histograms in parallel
        E.G. one for each (layer, neuron) pair, in which case tensor_shape=(n_layers, n_neurons)
        and self.update would be called with tensors of shape (n_layers, n_neurons, n_samples).
        '''
        
        self.min = min
        self.max = max
        self.bin_width = bin_width
        self.tensor_shape = tensor_shape
        
        self.boundaries = torch.arange(self.min, self.max+bin_width, bin_width)
        self.counts = torch.zeros(*tensor_shape, len(self.boundaries)-1).int()
    
    def update(self, obs):
        assert obs.shape[:-1] == self.tensor_shape
        obs = obs.detach().cpu()

        # flatten all but last dimension
        obs_view = obs.view(-1, obs.shape[-1])
        # bucket ids with shape (product(obs.shape[:-1]), obs.shape[-1])
        flattened_bucket_ids = torch.bucketize(obs_view.clamp(self.min, self.max-self.bin_width*1e-4), self.boundaries, right=True)-1
        self.counts += batched_bincount(flattened_bucket_ids, dim=-1, max_value=len(self.boundaries)-1).view(*obs.shape[:-1], -1)