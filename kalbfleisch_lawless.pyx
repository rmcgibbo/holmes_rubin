import numpy as np
import scipy.linalg
from numpy import zeros, asarray, ascontiguousarray, asfortranarray, real

from numpy cimport npy_intp
from libc.math cimport exp, sqrt, log
from libc.string cimport strcmp, memset
from libc.float cimport DBL_MIN, DBL_MAX

DEF DEBUG = False
include "cy_blas.pyx"
include "triu_utils.pyx"
include "_ratematrix_support.pyx"


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
        cdef double[::1] w, expwt
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
        cdef npy_intp n = self.n
        cdef npy_intp n_triu = self.n_triu
        cdef double t = self.t
        cdef double[::1] w
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


cpdef int build_ratemat(const double[::1] theta, npy_intp n, double[:, ::1] out,
                        const char* which=b'K'):
    r"""build_ratemat(theta, n, out, which='K')

    Build the reversible rate matrix K or symmetric rate matrix, S,
    from the free parameters, `\theta`

    Parameters
    ----------
    theta : array
        The free parameters, `\theta`. These values are the linearized elements
        of the upper triangular portion of the symmetric rate matrix, S,
        followed by the log equilibrium weights.
    n : int
        Dimension of the rate matrix, K, (number of states)
    which : {'S', 'K'}
        Whether to build the matrix S or the matrix K
    out : [output], array shape=(n, n)
        On exit, out contains the matrix K or S
    """
    cdef npy_intp u = 0, k = 0, i = 0, j = 0
    cdef npy_intp n_triu = n*(n-1)/2
    cdef double s_ij, K_ij, K_ji
    cdef double[::1] pi
    cdef int buildS = strcmp(which, 'S') == 0
    if DEBUG:
        assert out.shape[0] == n
        assert out.shape[1] == n
        assert theta.shape[0] == n_triu + n
        assert np.all(np.asarray(out) == 0)

    pi = zeros(n)
    for i in range(n):
        pi[i] = exp(theta[n_triu+i])

    for u in range(n_triu):
        k_to_ij(u, n, &i, &j)
        s_ij = theta[u]

        if DEBUG:
            assert 0 <= u < n*(n-1)/2

        K_ij = s_ij * sqrt(pi[j] / pi[i])
        K_ji = s_ij * sqrt(pi[i] / pi[j])
        if buildS:
           out[i, j] = s_ij
           out[j, i] = s_ij
        else:
            out[i, j] = K_ij
            out[j, i] = K_ji
        out[i, i] -= K_ij
        out[j, j] -= K_ji

    if DEBUG:
        assert np.allclose(np.array(out).sum(axis=1), 0.0)
        assert np.allclose(scipy.linalg.expm(np.array(out)).sum(axis=1), 1)
        assert np.all(0 < scipy.linalg.expm(np.array(out)))
        assert np.all(1 > scipy.linalg.expm(np.array(out)))

    return 0


cpdef double dK_dtheta_ij(const double[::1] theta, npy_intp n, npy_intp u,
                          double[:, ::1] A=None, double[:, ::1] out=None) nogil:
    r"""dK_dtheta_ij(theta, n, u, A=None, out=None)

    Compute :math:`dK_ij / dtheta_u` over all `i`, `j` for fixed `u`.

    Along with `dK_dtheta_u`, this function computes a slice of the 3-index
    tensor :math:`dK_ij / dtheta_u`, the derivative of the rate matrix `K`
    with respect to the free parameters,`\theta`. This function computes a 2D
    slice of this tensor over all (i,j) for a fixed `u`.

    Furthermore, this function _additionally_ makes it possible, using the
    argument `A`, to compute the hadamard product of this slice with a given
    matrix A directly.  Since dK/dtheta_u is a sparse matrix with a known
    sparsity structure, it's more efficient to just do the hadamard as we
    construct it, and never save the matrix elements directly.

    Parameters
    ----------
    theta : array
        The free parameters, `\theta`. These values are the linearized elements
        of the upper triangular portion of the symmetric rate matrix, S,
        followed by the log equilibrium weights.
    n : int
        Dimension of the rate matrix, K, (number of states)
    u : int
        The index, `0 <= u < len(theta)` of the element in `theta` to
        construct the derivative of the rate matrix, `K` with respect to.
    A : array of shape=(n, n), optional
        If not None, an arbitrary (n, n) matrix to be multiplied element-wise
        with the derivative of the rate matrix, dKu.
    out : [output], optional array of shape=(n, n)
        If not None, out will contain the matrix dKu on exit.

    Returns
    -------
    s : double
        The sum of the element-wise product of dK/du and A, if A is not None.
    """
    cdef npy_intp n_triu = n*(n-1)/2
    cdef npy_intp a, i, j
    cdef double dK_i, s_ij, dK_ij, dK_ji, pi_i, pi_j
    cdef double sum_elem_product = 0

    if DEBUG:
        assert out.shape[0] == n and out.shape[1] == n
        assert A.shape[0] == n and A.shape[1] == n
        assert theta.shape[0] == n_triu + n

    if out is not None:
        memset(&out[0,0], 0, n*n*sizeof(double))

    if u < n_triu:
        # the perturbation is to the triu rate matrix
        # first, use the linear index, u, to get the (i,j)
        # indices of the symmetric rate matrix
        k_to_ij(u, n, &i, &j)

        s_ij = theta[u]
        pi_i = exp(theta[n_triu+i])
        pi_j = exp(theta[n_triu+j])
        dK_ij = sqrt(pi_j / pi_i)
        dK_ji = sqrt(pi_i / pi_j)

        if A is not None:
            sum_elem_product = (
                A[i,j]*dK_ij + A[j,i]*dK_ji
              - A[i,i]*dK_ij - A[j,j]*dK_ji
            )

        if out is not None:
            out[i, j] = dK_ij
            out[j, i] = dK_ji
            out[i, i] -= dK_ij
            out[j, j] -= dK_ji

    else:
        # the perturbation is to the equilibrium distribution

        # `i` is now the index, in `pi`, of the perturbed element
        # of the equilibrium distribution.
        i = u - n_triu
        pi_i = exp(theta[n_triu+i])

        # the matrix dKu has 1 nonzero row, 1 column, and the diagonal. e.g:
        #
        #    x     x
        #      x   x
        #        x x
        #    x x x x x x
        #          x x
        #          x   x

        for j in range(n):
            if j == i:
                continue

            k = ij_to_k(i, j, n)
            s_ij = theta[k]
            pi_j = exp(theta[n_triu+j])
            dK_ij = -0.5 * s_ij * sqrt(pi_j / pi_i)
            dK_ji = 0.5  * s_ij * sqrt(pi_i / pi_j)

            if A is not None:
                sum_elem_product += (
                    A[i,j]*dK_ij + A[j,i]*dK_ji
                  - A[i,i]*dK_ij - A[j,j]*dK_ji
                )

            if out is not None:
                out[i, j] = dK_ij
                out[j, i] = dK_ji
                out[i, i] -= dK_ij
                out[j, j] -= dK_ji

    return sum_elem_product
