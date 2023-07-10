import numpy as np

class Problem:
    def __init__(self, func, bounds, true_minimum, x_min=None):
        self.func = func
        self.bounds = bounds
        self.true_minimum = true_minimum
        self.x_min = x_min
    
    @property
    def n_dim(self):
        return len(self.bounds)
    
    def __call__(self, x):
        return self.func(x)

    def __repr__(self) -> str:
        return self.func.__name__


def _styblinski_tang(x):
    """The Styblinski-Tang function is defined on the square
    :math:`x_i \\in [-5, 5]`.

    It has a global minimum at :math:`f(x*) = -39.1661657037714 * ndim` at
    x* = (-2.903534, ..., -2.903534)

    >>> _styblinski_tang([-2.903534] * 2)
    -78.3323314075428
    >>> _styblinski_tang([-2.903534] * 3)
    -117.4984971113142
    >>> _styblinski_tang([-2.903534] * 4)
    -156.6646628150856

    More details: <http://www.sfu.ca/~ssurjano/stybtang.html>
    """
    return 0.5 * np.sum(np.power(x,4) - 16 * np.square(x) + 5 * np.array(x))

n_dim = 2
# Each row is a dimension
bounds = [(-5.0, 5.0), ] * n_dim
x_min = np.array([-2.903534] * n_dim).reshape((n_dim, 1))
true_minimum = -39.16616570377142 * n_dim
f = _styblinski_tang
styblinski = Problem(f, bounds, true_minimum, x_min)


if __name__ == "__main__":
    import doctest
    doctest.testmod()