from anncore import create_network
from jax.random import PRNGKey

def test():
    key     = PRNGKey(0)
    network = create_network(key, 1000)
    assert len(network) == 1
    assert network[0].shape == (1000,1)
