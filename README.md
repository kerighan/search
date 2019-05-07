# Search

## Overview

This a simple Python class that allows to concurrently minimize or maximize arbitrary functions, by brute forcing a grid of parameters.

## Disclaimer

**This implementation has only two aims: ease and speed of use, not algorithm efficiency.**

## Dependencies

* numpy

## Tutorial

Simply instantiate a `Search` object, define the parameters space with the method `grid(**kwargs)` and use the method `minimize(function)` to minimize a given function.

```python
from search import Search
import numpy as np


# given data
x = np.linspace(0, 10, 100)
y = 3 * x + 2  # 3 and 2 are the parameters to infer


# loss function
def loss(x, y, a, b):
    y_pred = x * a + b
    loss = np.mean((y - y_pred)**2)
    return loss


search = Search(workers=4)  # 4 workers Pool
# `grid` requires iterable parameters,
# so giving [x] acts as one parameters `x`
search.grid(
    a=np.linspace(0, 11, 100),
    b=np.linspace(0, 11, 100),
    x=[x],
    y=[y]
)
data = search.minimize(loss)
print(data[0]["parameters"])  # a = 3, b = 2
```
