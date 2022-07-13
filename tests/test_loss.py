from jax.random import PRNGKey
from jax.numpy import array
from anncore import loss

def test():
    key      = PRNGKey(0)
    input    = array([[1.0]])
    expected = array([[2.0]])
    network  = [ array([[1.5]]) ]
    error    = loss(network, input, expected)
    assert error == 0.25
