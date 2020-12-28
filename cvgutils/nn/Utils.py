import numpy as np

def linearLR(x,iter0,iter1,lr0,lr1):
    return np.interp(x,[iter0,iter1],[lr0,lr1])