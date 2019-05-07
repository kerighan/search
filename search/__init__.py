from multiprocessing import Pool
from queue import Queue
from tqdm import tqdm
import numpy as np


def compute_process(items):
    """the atomic operation given to every parameters state."""
    func, x = items
    return {
        "loss": func(**x),
        "parameters": x
    }


class Search(object):
    def __init__(self, workers=4):
        self.workers = workers  # number of parallel workers
    
    def grid(self, **kwargs):
        """defines the complete grid of parameters."""
        self.dim = 1
        self.shape = []
        self.parameters_key = []
        self.parameters_value = []
        for key, value in kwargs.items():
            value_len = len(value)
            self.dim *= value_len
            self.shape.append(value_len)
            self.parameters_key.append(key)
            self.parameters_value.append(value)

    def get_parameters(self, i):
        """returns the `dict`:parameters from index `i`."""
        indices = np.unravel_index(i, self.shape)

        parameters = {}
        for key_idx, value_idx in enumerate(indices):
            key = self.parameters_key[key_idx]
            parameters[key] = self.parameters_value[key_idx][value_idx]
        return parameters
    
    def compute(self, func):
        """computes all values in the parameters space."""
        params = []
        for i in range(self.dim):
            params.append([func, self.get_parameters(i)])

        data = []
        pool = Pool(self.workers)
        for result in tqdm(
                pool.imap_unordered(compute_process, params),
                total=len(params),
                desc="Grid search"):
            data.append(result)
        return data
    
    def minimize(self, func):
        data = self.compute(func)
        data = sorted(data, key=lambda x: x['loss'])
        return data

    def maximize(self, func):
        data = self.compute(func)
        data = sorted(data, key=lambda x: x['loss'], reverse=True)
        return data
