from jax.numpy import array, array_equal
from anncore import right

def test():
    matrix = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    actual   = right(matrix)
    expected = array([
        [3],
        [6],
        [9]
    ])
    assert array_equal(actual, expected)