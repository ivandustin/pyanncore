from anncore import Network, feed, infer
from jax.numpy import array, isclose
from jax.random import PRNGKey
from jax.lax import fori_loop
from jax import jit

def test():
    key      = PRNGKey(0)
    network  = Network(key, 1)
    input    = array([[1.0]])
    expected = array([[1.7]])
    before   = infer(input, network)
    network  = train(network, input, expected)
    after    = infer(input, network)
    assert not isclose(before, expected)
    assert isclose(after, expected)

@jit
def train(network, input, expected):
    body = lambda i, network: feed(input, expected, network, 0.1)
    return fori_loop(0, 50, body, network)
