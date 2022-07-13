from annfunctions import optimizer, prune
from jax.tree_util import tree_map
from .loss import loss
from jax import grad

def feed(input, expected, network, learning_rate):
    gradient = grad(loss)(network, input, expected)
    return tree_map(lambda synapse, gradient: prune(optimizer(synapse, gradient, learning_rate)), network, gradient)
