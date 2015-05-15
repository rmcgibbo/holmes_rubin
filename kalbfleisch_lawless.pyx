# cython: boundscheck=False, cdivision=True, wraparound=False, c_string_encoding=ascii
import numpy as np
import scipy.linalg
import scipy.optimize
from numpy import zeros, asarray, ascontiguousarray, asfortranarray, real

from numpy cimport npy_intp
from libc.math cimport exp, sqrt, log
from libc.string cimport strcmp, memset
from libc.float cimport DBL_MIN, DBL_MAX

include "support.pyx"


cdef class KalbfleischLawless(object):
    cdef double t
    cdef npy_intp n, n_triu
    cdef double[:, ::1] countsmat

    def __cinit__(self, const double[:, ::1] countsmat, double t=1):
        if countsmat.shape[0] != countsmat.shape[0]:
            raise ValueError('countsmat must be square')

        self.countsmat = countsmat
        self.t = t
        self.n = countsmat.shape[0]
        self.n_triu = self.n * (self.n-1)/2

    def fit(self, const double[::1] theta, int max_iter=100):
        result = scipy.optimize.minimize(
            self.f, jac=self.fprime, hess=self.fhess, x0=theta,
            options=dict(maxiter=max_iter), method='trust-ncg')
        return result

    def f(self, const double[::1] theta):
        cdef npy_intp i, j
        cdef npy_intp n = self.n
        cdef npy_intp n_triu = self.n_triu
        cdef double logl
        cdef double t = self.t
        cdef double[::1] w, expwt, pi
        cdef double[:, ::1] countsmat, T
        countsmat = self.countsmat

        _, _, _, T = self._transmat(theta)

        logl = 0
        for i in range(n):
            for j in range(n):
                if countsmat[i, j] > 0:
                    logl += countsmat[i, j] * log(T[i, j])

        return logl

    def fprime(self, const double[::1] theta):
        cdef npy_intp u, i, j
        cdef npy_intp n = self.n
        cdef double t = self.t
        cdef npy_intp size = theta.shape[0]
        cdef double[::1] grad, w, pi, expwt
        cdef double[:, ::1] U, V, dT_dTheta, countsmat, T
        countsmat = self.countsmat
        grad = zeros(size)
        expwt = zeros(n)
        dKu = zeros((n, n))

        w, U, V, T = self._transmat(theta)
        for i in range(n):
            expwt[i] = exp(w[i]*t)

        for u in range(size):
            dT_dTheta =  self._dT_dTheta(theta, u, w, expwt, U, V)

            for i in range(n):
                for j in range(n):
                    if countsmat[i, j] > 0:
                        grad[u] += (countsmat[i, j] * dT_dTheta[i, j] / T[i, j])

        return ascontiguousarray(grad)

    def fhess(self, const double[::1] theta):
        cdef npy_intp u, i, j
        cdef npy_intp n = self.n
        cdef npy_intp size = theta.shape[0]
        cdef double hessian_ab
        cdef double t = self.t
        cdef double[::1] w, expwt, rowsums
        cdef double[:, ::1] U, V, T, Q, dT_dTheta_a, dT_dTheta_b, hessian
        expwt = zeros(n)
        Q = zeros((n, n))
        hessian = zeros((size, size))
        rowsums = np.sum(self.countsmat, axis=1)

        w, U, V, T = self._transmat(theta)
        for i in range(n):
            expwt[i] = exp(w[i]*t)

        for i in range(n):
            for j in range(n):
                Q[i,j] = -rowsums[i] / T[i, j]

        for a in range(size):
            dT_dTheta_a =  self._dT_dTheta(theta, a, w, expwt, U, V)
            for b in range(a, size):
                dT_dTheta_b =  self._dT_dTheta(theta, b, w, expwt, U, V)

                hessian_ab = 0
                for i in range(n):
                    for j in range(n):
                        hessian_ab += Q[i, j] * dT_dTheta_a[i, j]  * dT_dTheta_b[i, j]

                hessian[a, b] = hessian_ab
                hessian[b, a] = hessian_ab

        return asarray(hessian)

    def _transmat(self, double[::1] theta):
        cdef npy_intp i
        cdef npy_intp n = self.n
        cdef npy_intp n_triu = self.n_triu
        cdef double t = self.t
        cdef double rowsum
        cdef double[::1] w, pi
        cdef double[:, ::1] U, V, S, T, temp1
        S = zeros((n, n))
        T = zeros((n, n))
        temp1 = zeros((n, n))
        expwt = zeros(n)
        pi = zeros(n)

        build_ratemat(theta, n, S, 'S')

        for i in range(n):
            pi[i] = exp(theta[n_triu+i])

        w, U, V = eig_K(S, n, pi, 'S')
        for i in range(n):
            expwt[i] = exp(w[i]*t)

        transmat(expwt, U, V, n, temp1, T)

        for i in range(n):
            rowsum = 0
            for j in range(n):
                if T[i, j] <= 0.0:
                    T[i, j] = 1e-20
                rowsum += T[i, j]
            for j in range(n):
                T[i, j] = T[i, j] / rowsum
        return w, U, V, T

    cdef double[:, ::1] _dT_dTheta(self, double[::1] theta, npy_intp u, double[::1] w,
                                   double[::1] expwt, double[:, ::1] U, double[:, ::1] V):
        cdef npy_intp n = self.n
        cdef double t = self.t
        cdef double[:, ::1] dKu, temp1, temp2

        dKu = zeros((n, n))
        temp1 = zeros((n, n))
        temp2 = zeros((n, n))

        dK_dtheta_ij(theta, n, u, A=None, out=dKu)
        cdgemm_TN(U, dKu, temp1)
        cdgemm_NN(temp1, V, temp2)
        hadamard_X(w, expwt, t, n, temp2)
        cdgemm_NN(V, temp2, temp1)
        cdgemm_NT(temp1, U, temp2)

        return temp2

