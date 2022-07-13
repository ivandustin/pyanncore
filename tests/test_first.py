from jax.numpy import array, array_equal
from anncore import first

def test():
    matrix = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    actual   = first(matrix)
    expected = array([
        [1],
        [4],
        [7]
    ])
    assert array_equal(actual, expected)
