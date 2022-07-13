from anncore import Network, neurogenesis
from jax.random import PRNGKey, split

def test_b():
    key     = PRNGKey(0)
    keys    = split(key, 6)
    network = Network(key, 100)
    for key in keys:
        network = neurogenesis(key, network)
    assert len(network) == 7
    assert network[0].shape == (100, 7)
    assert network[1].shape == (1, 6)
    assert network[2].shape == (1, 5)
    assert network[3].shape == (1, 4)
    assert network[4].shape == (1, 3)
    assert network[5].shape == (1, 2)
    assert network[6].shape == (1, 1)
