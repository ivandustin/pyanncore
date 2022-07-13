from annfunctions import loss as loss_function
from .infer import infer

def loss(network, input, expected):
    predicted = infer(input, network)
    if callable(expected):
        expected = expected(predicted)
    return loss_function(expected, predicted)
