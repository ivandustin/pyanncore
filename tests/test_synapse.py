from anncore import Synapse
from jax.random import PRNGKey
from jax.numpy import isclose

def test():
    key     = PRNGKey(0)
    synapse = Synapse(key, shape=(1000,))
    assert isclose(synapse.mean(), 1.5, atol=0.01)
    assert isclose(synapse.std(),  0.1, atol=0.01)
