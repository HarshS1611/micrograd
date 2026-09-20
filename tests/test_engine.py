"""
Test suite for engine.py — run with:
    cd micrograd_project && pytest tests/ -v

Run just the fast/basic ones first:
    pytest tests/test_engine.py -v -k "not golden"

Each test checks one specific behaviour; the test name and docstring
identify which _backward closure (or which part of backward()) it covers.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine import Value


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ------------------------------------------------------------
# Forward pass — do the ops compute the right VALUE at all?
# ------------------------------------------------------------

def test_add_forward():
    a, b = Value(2.0), Value(3.0)
    assert approx((a + b).data, 5.0)


def test_add_with_plain_number():
    a = Value(2.0)
    assert approx((a + 3).data, 5.0)
    assert approx((3 + a).data, 5.0)  # exercises __radd__


def test_mul_forward():
    a, b = Value(2.0), Value(3.0)
    assert approx((a * b).data, 6.0)


def test_mul_with_plain_number():
    a = Value(2.0)
    assert approx((a * 3).data, 6.0)
    assert approx((3 * a).data, 6.0)  # exercises __rmul__


def test_pow_forward():
    a = Value(3.0)
    assert approx((a ** 2).data, 9.0)


def test_sub_forward():
    a, b = Value(5.0), Value(3.0)
    assert approx((a - b).data, 2.0)


def test_truediv_forward():
    a, b = Value(6.0), Value(3.0)
    assert approx((a / b).data, 2.0)


def test_relu_forward():
    assert approx(Value(5.0).relu().data, 5.0)
    assert approx(Value(-5.0).relu().data, 0.0)


def test_tanh_forward():
    # tanh(0) == 0 exactly
    assert approx(Value(0.0).tanh().data, 0.0)
    # tanh saturates toward 1 for large positive input
    assert Value(10.0).tanh().data > 0.999


# ------------------------------------------------------------
# Backward pass — the actual point. Each of these checks a specific
# hand-computable derivative.
# ------------------------------------------------------------

def test_add_backward():
    a, b = Value(2.0), Value(3.0)
    c = a + b
    c.backward()
    # d(a+b)/da = 1, d(a+b)/db = 1
    assert approx(a.grad, 1.0)
    assert approx(b.grad, 1.0)


def test_mul_backward():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    c.backward()
    # d(a*b)/da = b, d(a*b)/db = a
    assert approx(a.grad, 3.0)
    assert approx(b.grad, 2.0)


def test_pow_backward():
    a = Value(3.0)
    b = a ** 2
    b.backward()
    # d(a^2)/da = 2*a = 6
    assert approx(a.grad, 6.0)


def test_relu_backward_positive():
    a = Value(3.0)
    out = a.relu()
    out.backward()
    assert approx(a.grad, 1.0)  # gradient passes through unchanged


def test_relu_backward_negative():
    a = Value(-3.0)
    out = a.relu()
    out.backward()
    assert approx(a.grad, 0.0)  # gradient is killed


def test_multiuse_gradient_accumulates():
    """THE classic autograd bug: if a Value is used twice, its two
    gradient contributions must ADD, not overwrite. y = x + x means
    dy/dx = 2, not 1. A failure here means a _backward used `=` instead of `+=`
    somewhere in a _backward closure.
    """
    x = Value(3.0)
    y = x + x
    y.backward()
    assert approx(x.grad, 2.0), (
        f"expected x.grad == 2.0 (used '+=' correctly), got {x.grad} "
        "— check every _backward closure uses += , never ="
    )


def test_multiuse_in_multiplication():
    """Same bug, via multiplication: y = x * x means dy/dx = 2x."""
    x = Value(4.0)
    y = x * x
    y.backward()
    assert approx(x.grad, 8.0)


# ------------------------------------------------------------
# Gradient check via finite differences — same technique as
# a standard gradient check, applied to THIS engine.
# If the analytic backward() is right, it should always agree with
# the numerical estimate, for ANY expression, not just the hand-picked
# cases above. This is the strongest test in this file.
# ------------------------------------------------------------

def numerical_grad(f, x0, eps=1e-6):
    return (f(x0 + eps) - f(x0 - eps)) / (2 * eps)


def test_gradcheck_compound_expression():
    def f(x):
        a = Value(x)
        b = Value(-2.0)
        c = a * b + b ** 2
        d = c.relu() + a
        return d.data

    x0 = 3.0
    numeric = numerical_grad(f, x0)

    a = Value(x0)
    b = Value(-2.0)
    c = a * b + b ** 2
    d = c.relu() + a
    d.backward()

    assert approx(a.grad, numeric, tol=1e-4), (
        f"analytic grad {a.grad} vs numerical grad {numeric} — "
        "these should match closely for ANY valid backward() implementation"
    )


# ------------------------------------------------------------
# THE GOLDEN TEST — Karpathy's own README example. These exact
# numbers are cross-checked against PyTorch (I additionally verified
# them independently with finite differences before writing this
# test). If this passes, the engine is correct.
# ------------------------------------------------------------

def test_golden_karpathy_example():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()

    assert approx(g.data, 24.70408163265306, tol=1e-4)
    assert approx(a.grad, 138.83381926049765, tol=1e-2)
    assert approx(b.grad, 645.5772595348463, tol=1e-2)
