from jax.numpy import array, array_equal
from anncore import infer

def test():
    x = array([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    S = [
        array([
            [1.0, 0.1],
            [0.2, 1.0]
        ]),
        array([
            [0.5]
        ])
    ]
    expected = array([
        [1.7],
        [0.0],
        [1.0],
        [0.0]
    ])
    assert array_equal(infer(x, S), expected)
