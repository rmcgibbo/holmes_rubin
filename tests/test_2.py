import numpy as np
import scipy.linalg
from scipy.optimize import check_grad
from msmbuilder.msm import _ratematrix
from kalbfleisch_lawless import KalbfleischLawless
random = np.random.RandomState(0)

# countsmat1 = np.array(
#       [[  7.338e+03,   0.000e+00,   3.040e+02,   1.100e+01,   1.700e+01,
#           2.961e+03,   1.300e+01,   2.497e+03],
#        [  1.000e+00,   6.399e+03,   0.000e+00,   3.920e+02,   3.383e+03,
#           9.100e+01,   2.476e+03,   3.000e+00],
#        [  3.100e+02,   0.000e+00,   1.441e+04,   5.900e+01,   4.000e+00,
#           2.200e+01,   0.000e+00,   3.482e+03],
#        [  9.000e+00,   3.790e+02,   5.500e+01,   4.838e+03,   2.207e+03,
#           2.400e+01,   3.270e+02,   6.500e+01],
#        [  2.400e+01,   3.377e+03,   4.000e+00,   2.145e+03,   6.182e+03,
#           7.800e+01,   3.408e+03,   3.900e+01],
#        [  2.972e+03,   8.600e+01,   3.100e+01,   1.400e+01,   6.800e+01,
#           4.181e+03,   6.600e+01,   6.170e+02],
#        [  1.500e+01,   2.495e+03,   0.000e+00,   3.590e+02,   3.354e+03,
#           5.900e+01,   6.869e+03,   1.300e+01],
#        [  2.475e+03,   9.000e+00,   3.474e+03,   8.700e+01,   4.300e+01,
#           6.210e+02,   6.000e+00,   4.742e+03]])


countsmat1 = np.array([
    [10.0, 3.0, 0.0,],
    [0.0, 5.0, 2.0],
    [0.0, 3.0, 8.0]
])

def example_theta(n):
    return random.uniform(0, 1, size=int(n*(n-1)/2 + n))


def test_1():
    n = countsmat1.shape[0]
    theta = example_theta(n)

    fitter = KalbfleischLawless(countsmat1, t=1)
    assert check_grad(fitter.f, fitter.fprime, theta) < 1e-6


def test_2():
    n = countsmat1.shape[0]
    theta = example_theta(n)

    fitter = KalbfleischLawless(countsmat1, t=1)
    logl = fitter.f(theta)
    grad = fitter.fprime(theta)

    logl2, grad2 = _ratematrix.loglikelihood(theta, countsmat1, t=1)

    np.testing.assert_almost_equal(logl, logl2)
    np.testing.assert_array_almost_equal(grad, grad2)


def test_3():
    n = countsmat1.shape[0]
    theta = example_theta(n)

    fitter = KalbfleischLawless(countsmat1, t=1)
    logl = fitter.f(theta)
    grad = fitter.fprime(theta)
    hess = fitter.fhess(theta)

    hess2 = _ratematrix.hessian(theta, countsmat1, t=1)

    np.testing.assert_array_almost_equal(hess, hess2)


def test_4():
    n = countsmat1.shape[0]
    theta = example_theta(n)

    fitter = KalbfleischLawless(countsmat1, t=1)
    print(fitter.fit(theta))

