# cython: boundscheck=False, cdivision=True, wraparound=False, c_string_encoding=ascii
import numpy as np
import scipy.linalg
from numpy import zeros, ascontiguousarray, asfortranarray, real

from numpy cimport npy_intp
from libc.math cimport sqrt, exp, log
from libc.string cimport memset, strcmp
from libc.float cimport DBL_MIN, DBL_MAX
cdef double log_dbl_min = log(DBL_MIN)
cdef double log_dbl_max = log(DBL_MAX)
cdef double NAN = np.nan
DEF DEBUG = False

include "cy_blas.pyx"
include "triu_utils.pyx"


cpdef eig_K(const double[:, ::1] A, npy_intp n, double[::1] pi=None, which='K'):
    r"""eig_K(A, n, pi=None, which='K')

    Diagonalize the rate matrix, K, from either the matrix K or the symmetric
    rate matrix, S.

    If which == 'K', the first argument should be the rate matrix, K, and `pi`
    is ignored. If which == 'S', the first argument should be the symmetric
    rate matrix, S. This can be build using buildK(... which='S'), and pi
    should contain the equilibrium distribution (left eigenvector of K with
    eigenvalue 0, and also the last n elements of exptheta).

    Whichever is supplied the return value is the eigen decomposition of `K`.
    The eigendecomposition of S is not returned.

    Using the symmetric rate matrix, S, is somewhat faster and more numerically
    stable.

    Returns
    -------
    w : array
        The eigenvalues of K
    U : array, size=(n,n)
        The left eigenvectors of K
    V : array, size=(n,n)
        The right eigenvectors of K
    """
    cdef npy_intp i, j
    cdef double norm = 1
    cdef double[::1] w
    cdef double[::1, :] U, V, VS
    U = zeros((n, n), order='F')
    V = zeros((n, n), order='F')

    if which == 'S':
        w, VS = scipy.linalg.eigh(A)
        with nogil:
            for j in range(n):
                for i in range(n):
                    V[i, j] = sqrt(pi[j] / pi[i]) * VS[i, j]
                    U[i, j] = sqrt(pi[i] / pi[j]) * VS[i, j]
            for i in range(n):
                cdnrm2(V[:, i], &norm)
                for j in range(n):
                    V[j, i] /= norm
                    U[j, i] *= norm

    else:
        w_, U_, V_ = scipy.linalg.eig(A, left=True, right=True)
        w = ascontiguousarray(real(w_))
        U = asfortranarray(real(U_))
        V = asfortranarray(real(V_))

        with nogil:
            for i in range(n):
                # we need to ensure the proper normalization
                cddot(U[:, i], V[:, i], &norm)
                for j in range(n):
                    U[j, i] = U[j, i] / norm

    if DEBUG:
        assert np.allclose(scipy.linalg.inv(V).T, U)

    return w, U.copy(), V.copy()


cdef int hadamard_X(const double[::1] w, const double[::1] expwt, double t,
                    npy_intp n, double[:, ::1] A) nogil:
    """
    Overwrite the matrix A by the elementwise product of A with the matrix
    X, where :math:`x_{ij}` is:

        if i != j:
            x_{ij} = (e^{t w_i} - e^{t w_j}) / (w_i - w_j)
        else:
            x_{ii} = t * e^{t w_i}

    :math:`x_{ij}` is computed in a more numerically stable way (if w_1 ~ w_2)
    as ::

      x_{ij} = (e^{t w_1 - t w_2} - 1) / (t w_1 - t w_2)  * t e^{t w_2)

    using a special numerically stable routine for ``exprel(x) = (exp(x)-1)/x``.
    """
    cdef npy_intp i, j
    cdef double X_ij

    for i in range(n):
        for j in range(n):
            if i != j:
                #X_ij = (expwt[i] - expwt[j]) / (w[i] - w[j])
                X_ij = exprel(t*(w[i]-w[j])) * t * expwt[j]
                A[i, j] *= X_ij
            else:
                A[i, j] *= t * expwt[i]


cdef int hadamard_inplace(const double[:, ::1] A, const double[:, ::1] B) nogil:
    """Overwrite the matrix A by its element-wise product with matrix B
    """
    if (A.shape[0] != B.shape[0]) or (A.shape[1] != B.shape[1]):
        return -1

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = A[i, j] * B[i, j]

    return 1


cdef int transmat(const double[::1] expwt, const double[:, ::1] U,
                  const double[:, ::1] V, npy_intp n,
                  double[:, ::1] temp, double[:, ::1] T) nogil:
    """Compute the transition matrix, expm(Kt), from the eigen-decomposition
    of K

    On exit, T is written into the variable `T`. temp is an n x n workspace.
    """
    cdef npy_intp i, j
    cdef double rowsum
    # T = np.dot(np.dot(V, np.diag(expwt)), U.T)
    for i in range(n):
        for j in range(n):
            temp[i, j] = V[i, j] * expwt[j]
    cdgemm_NT(temp, U, T)


cpdef int dw_du(const double[:, ::1] dKu, const double[:, ::1] U,
            const double[:, ::1] V, npy_intp n, double[::1] temp,
            double[::1] out) nogil:
    r"""Calculate the derivative of the eigenvalues, w, of a matrix, K(\theta),
    with respect to \theta_u.

    Parameters
    ----------
    dKu : array, shape=(n, n)
        Derivative of the rate matrix, K(\theta), with respect to \theta_u
    U : array, shape=(n, n)
        Left eigenvectors of the rate matrix, K(\theta)
    V : array, shape=(n, n)
        Right eigenvectors of the rate matrix, K(\theta)
    n : int
        Size of the matrices
    temp : array, shape=(n,)
        Temporary storage (overwritten)

    Returns
    -------
    out : array, shape=(n,)
        On exit, out[i] contains the derivative of the `i`th eigenvalue
        of K with respect to \theta_u.
    """
    cdef npy_intp i
    for i in range(n):
        cdgemv_N(dKu, V[:, i], temp)
        cddot(temp, U[:, i], &out[i])



cdef dT_dtheta(const double[::1] w, const double[:, ::1] U, const double[:, ::1] V,
               const double[:, ::1] counts, npy_intp n, double t, double[:, ::1] T,
               double[:, ::1] dT):
    """Compute some of the terms required for d(exp(K))/d(theta).

    Returns
    -------
    """
    cdef double rowsum
    cdef npy_intp i, j
    cdef double[::1] expwt
    cdef double[:, ::1] X, temp1, temp2, dLdK
    temp1 = zeros((n, n))
    temp2 = zeros((n, n))
    dLdK = zeros((n, n))
    expwt = zeros(n)

    with nogil:
        for i in range(n):
            expwt[i] = exp(w[i]*t)

        transmat(expwt, U, V, n, temp1, T)

        # When the rate matrix is not irreducible, T contains zeros,
        # which messes things up
        for i in range(n):
            rowsum = 0
            for j in range(n):
                if T[i, j] <= 0.0:
                    T[i, j] = 1e-20
                rowsum += T[i, j]
            for j in range(n):
                T[i, j] = T[i, j] / rowsum

        # dLdK[i,j] = counts[i,j] / T[i,j]
        for i in range(n):
            for j in range(n):
                if counts[i, j] > 0 and T[j, j] > 0:
                    dLdK[i, j] = counts[i, j] / T[i, j]

        # out = U \left(V^T dLdK U \circ X(\lambda, t))\right) V^T

        # temp2 = V^T dLdK U
        cdgemm_TN(V, dLdK, temp1)
        cdgemm_NN(temp1, U, temp2)

        # temp2 =  (V^T dLdK U \circ X(w, t))
        hadamard_X(w, expwt, t, n, temp2)

        # dT = U \left(V^TCU \circ X(\lambda, t))\right) V^T
        cdgemm_NN(U, temp2, temp1)
        cdgemm_NT(temp1, V, dT)

    if DEBUG:
        X = np.subtract.outer(expwt, expwt) / np.subtract.outer(w, w)
        np.fill_diagonal(np.asarray(X), t*np.asarray(expwt))
        Y = np.asarray(U).dot(np.asarray(V).T.dot(dLdK).dot(U) * X).dot(np.asarray(V).T)

        assert np.allclose(dT, Y)
        assert np.allclose(T, np.dot(np.dot(V, np.diag(expwt)), U.T))
        assert np.allclose(T, scipy.linalg.expm(t*np.asarray(K)))


cdef double exprel(double x) nogil:
    """Compute the quantity (\exp(x)-1)/x using an algorithm that is accurate
    for small x.
    """

    cdef double cut = 0.002;

    if (x < log_dbl_min):
        return -1.0 / x;

    if (x < -cut):
      return (exp(x) - 1.0)/x

    if (x < cut):
      return (1.0 + 0.5*x*(1.0 + x/3.0*(1.0 + 0.25*x*(1.0 + 0.2*x))))

    if (x < log_dbl_max):
      return (exp(x) - 1.0)/x

    return NAN



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
