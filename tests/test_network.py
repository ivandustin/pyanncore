from jax.random import PRNGKey
from anncore import Network

def test():
    key     = PRNGKey(0)
    network = Network(key, 1000)
    assert len(network) == 1
    assert network[0].shape == (1000, 1)
