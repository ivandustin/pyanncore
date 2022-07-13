from anncore import create_network, neurogenesis
from jax.random import PRNGKey, split
from jax.numpy import array_equal

def test_a():
    key     = PRNGKey(0)
    network = create_network(key, 100)
    network = neurogenesis(key, network)
    assert len(network) == 2
    assert network[0].shape == (100,2)
    assert network[1].shape == (1,1)

def test_b():
    key     = PRNGKey(0)
    network = create_network(key, 100)
    network = neurogenesis(key, network)
    network = neurogenesis(key, network)
    assert len(network) == 3
    assert network[0].shape == (100,3)
    assert network[1].shape == (1,2)
    assert network[2].shape == (1,1)

def test_c():
    key = PRNGKey(0)
    key_a, key_b = split(key)
    network_a = create_network(key_a, 100)
    network_b = neurogenesis(key_b, network_a)
    assert array_equal(network_a[0][:,0], network_b[0][:,-1])

def test_d():
    key = PRNGKey(0)
    key_a, key_b = split(key)
    network   = create_network(key, 100)
    network_a = neurogenesis(key_a, network)
    network_b = neurogenesis(key_b, network_a)
    assert array_equal(network_a[0][:,0],  network_b[0][:,0])
    assert array_equal(network_a[0][:,-1], network_b[0][:,-1])
