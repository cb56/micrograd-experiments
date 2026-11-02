import pytest
import math
from micrograd.autograd import Value

def test_addition_with_scalar_both_orders():
    aData = 3.0
    bData = 5.0
    
    a = Value(aData)
    b = Value(bData)

    assert(a + b).data == (b + a).data == 8.0

def test_multiplication_with_scalar_both_orders():
    aData = 3.0
    bData = 5.0
    
    a = Value(aData)
    b = Value(bData)

    assert(a * b).data == (b * a).data == 15.0

@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_tanh_gradient(x):
    a = Value(x)

    result = a.tanh()
    result.backward()

    expected = 1 - math.tanh(x) ** 2

    assert a.grad == pytest.approx(expected)