from annfunctions import loss as loss_function
from .infer import infer

def loss(network, input, observed):
    predicted = infer(input, network)
    if callable(observed):
        observed = observed(predicted)
    return loss_function(observed, predicted)
