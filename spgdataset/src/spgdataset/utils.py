def exact_div(x, y):
    assert x % y == 0
    return x // y


def exact_mult(x, y):
    assert int(x * y) == x * y
    return int(x * y)
