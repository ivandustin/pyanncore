from annfunctions import activation

def infer(x, S):
    s = S[0]
    n = x @ s
    x = activation(n[:,:1])
    for i in range(1, len(S)):
        s = S[i]
        n = n[:,1:]
        n = (x @ s) + n
        x = activation(n[:,:1])
    return x
