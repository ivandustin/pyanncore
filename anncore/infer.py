from annfunctions import activation
from .first import first
from .tail import tail

def infer(x, S):
    n = 0
    for s in S:
        n = (x @ s) + n
        x = activation(first(n))
        n = tail(n)
    return x
