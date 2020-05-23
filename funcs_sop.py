"""
This module contains a class with the definition of the SOP-FBR function
and its corresponding gradients, both with respect to the variables
and the parameters. The first one of these gradients is useful to
obtain the gradients of the function, whereas the second one comes very
handy in the optimization of the parameters.
"""
import time
import numpy as np
from numpy.polynomial import chebyshev as cheby
from scipy.optimize import approx_fprime as grad_diff
import tensorly as tl

# Define a fancy timer class decorator


class ArtDeco:
    """
    Contains a timer decorator
    Use inside other classes defined in this module as:
    @ArtDeco.timethis
    """
    @classmethod
    def timethis(cls, meth):
        """Decorator to provided execution time"""
        def timed(*args, **kw):
            t_0 = time.time()
            res = meth(*args, **kw)
            t_f = time.time()
            print(f"Method {meth.__name__} executed in {t_f - t_0} s")
            return res
        return timed

# Define the SopFunc class


class SopFbr:
    """
    Class that holds the SOP-FBR function and its gradients
    with respect to the variables and parameters.

    Parameters
    ==========

    chebdim : int
            Dimension of the Chebyshev polynomial series. Assumes
            series of the same degree.
    gdim : array
         Array containing the shape of the Core tensor
    carray : array
           Array containing the parameter set. Concatenates the Chebyshev
           series coefficients with the core tensor, both flattened. The
           order of the Chebyshev coefficients is given by the order of the
           DOFs in the core tensor.
    """

    def __init__(self, chebdim, gdim, carray):
        self.chebdim = chebdim
        self.gdim = gdim
        self.carray = carray

        # Total number of Chebyshev coefficients
        self.ncheb = np.sum(self.gdim) * self.chebdim

        # Creates the Chebyshev coefficient's tensor
        self._chebs_tk = np.array(
            np.split(self.carray[:self.ncheb],
                     self.carray[:self.ncheb].shape[0] / self.chebdim))
        self.chebs = np.array(
            np.split(self._chebs_tk, np.cumsum(self.gdim))[0:-1])

        # Creates the core tensor
        self.cora = self.carray[self.ncheb:].reshape(self.gdim)

    def sop_fun(self, q_array):
        """
        Computes the value of the SOP-FBR potential by first
        conforming the vij(k) matrices, then reshaping
        the core tensor, and performing the tensor n-mode product.

        Parameters
        ==========

        q_array : array
                 Array of the values of the DVR in each DOFs

        Returns
        =======

        prod : float or array
             The values of the SOP-FBR in a point or in a grid
        """
        # Generates the matrices (or vectors) of the SPPs
        v_matrices = []
        for kdof, m_kp in enumerate(self.gdim):
            v_kp = np.zeros(m_kp)
            for j_kp in np.arange(m_kp):
                v_kp[j_kp] = cheby.chebval(
                    q_array[kdof], self.chebs[kdof][j_kp])
            v_matrices.append(v_kp)
        v_matrices = np.array(v_matrices)

        # Tensor n-mode product of the Tucker tensor
        prod = tl.tucker_tensor.tucker_to_tensor((self.cora, v_matrices))

        return prod

    @ArtDeco.timethis
    def sop_vargrad(self, q_array):
        """
        Computes the gradient of the SOPFBR function with respect
        to the variables

        Parameters
        ==========

        q_array : array
                 Array of the values of the DVR in each DOFs

        Returns
        =======

        vargrad : array
                Variable's gradient in the selected point
        """
        vargrad = np.zeros(np.shape(q_array))

        # Generates the matrices (or vectors) of the SPPs and derivatives
        v_matrices = []
        v_derivate = []
        for kdof, m_kp in enumerate(self.gdim):
            v_kp = np.zeros(m_kp)
            v_kp_der = np.zeros(m_kp)
            for j_kp in np.arange(m_kp):
                v_kp[j_kp] = cheby.chebval(
                    q_array[kdof], self.chebs[kdof][j_kp])
                v_kp_der[j_kp] = cheby.chebval(
                    q_array[kdof], cheby.chebder(self.chebs[kdof][j_kp]))
            v_matrices.append(v_kp)
            v_derivate.append(v_kp_der)
        v_matrices = np.array(v_matrices)
        v_derivate = np.array(v_derivate)

        # Calculate gradient components

        for kdof, _ in enumerate(q_array):
            matrices = np.copy(v_matrices)
            matrices[kdof] = v_derivate[kdof]
            vargrad[kdof] = tl.tucker_tensor.tucker_to_tensor(
                (self.cora, matrices))

        return vargrad

    def sop_pargrad(self):
        """
        Computes the gradient of the SOPFBR function with respect
        to the parameters

        Parameters
        ==========

        q_array : array
                 Array of the values of the DVR in each DOFs

        Returns
        =======

        pargrad : array
                Parameter's gradient in the selected point
        """


if __name__ == "__main__":
    CHEBDIM = 7
    GDIM = np.array([5, 5, 5, 5, 5, 5])
    CARRAY = np.loadtxt('params_init')
    QARR = np.array([2.6, 1.8, 2.2, 1.7, 1.9, np.pi])

    sop_hono = SopFbr(CHEBDIM, GDIM, CARRAY)
    print(sop_hono.sop_fun(QARR))
    print(sop_hono.sop_vargrad(QARR))
    T0 = time.time()
    grad_diff(QARR, sop_hono.sop_fun, 1e-7)
    TF = time.time()
    print(f"Nurical gradient in {TF - T0} s")
    print(grad_diff(QARR, sop_hono.sop_fun, 1e-7))
