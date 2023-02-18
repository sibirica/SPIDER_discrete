import numpy as np
from numpy import polynomial as p
from numbers import Real


def weight_leg(q: int) -> p.Polynomial:
    """
    Returns qth Legendre polynomial as a polynomial object.
    :param q: order of Legendre polynomial.
    :return: qth Legendre polynomial as a polynomial object.
    """
    orders = np.zeros(q + 1)
    orders[q] = 1
    leg = p.legendre.leg2poly(orders)
    poly = p.Polynomial(leg)
    return poly


def weight_poly(m: int) -> p.Polynomial:
    """
    Returns (x^2-1)^m as a polynomial object.
    :param m: Power of polynomial.
    :return: (x^2-1)^m as a polynomial object.
    """
    return p.Polynomial([-1, 0, 1]) ** m


def weight_1d(m: int, q: int, k: int, dx: Real) -> p.Polynomial:
    """
    Returns the kth derivative of the 1D weight function.
    :param m: Power of (x^2-1) polynomial.
    :param q: Order of Legendre polynomial.
    :param k: Order of derivative.
    :param dx: Grid spacing.
    :return: kth derivative of the 1D weight function.
    """
    poly = weight_leg(q) * weight_poly(m)
    poly = poly / dx ** k
    return poly.deriv(k)
