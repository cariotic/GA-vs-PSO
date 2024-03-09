import numpy as np


def g(x1, x2):
    y = np.exp(-(x1 ** 2 + x2 ** 2))
    return y

def fitness_measure(x1, x2):
    z = np.sqrt(x1 ** 2 + x2 ** 2)
    y = x1**2 + 5*x2**2 + 5*z*(np.sin(6 * np.arctan2(x2, x1)+5*z) ** 3) - 100*g(x1+3, x2+3) - 125*g(x1-2, x2-2) - np.exp(x1 * .005)
    return y