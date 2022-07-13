from anncore import Network, neurogenesis, feed, infer
from jax.numpy import array, array_equal, clip
from jax.random import PRNGKey, split
from jax.lax import fori_loop
from jax import jit

def test():
    key = PRNGKey(0)
    key_a, key_b = split(key)
    network = Network(key_a, 2)
    network = neurogenesis(key_b, network)
    input   = array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    expected = array([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ])
    network = train(network, input, expected)
    assert array_equal(infer(input, network).clip(0, 1), expected)

@jit
def train(network, input, expected):
    body = lambda i, network: feed(input, expected, network, 0.1)
    return fori_loop(0, 30, body, network)
