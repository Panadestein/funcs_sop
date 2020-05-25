"""
This module contains a class with the definition of the SOP-FBR function
and its corresponding gradients, both with respect to the variables
and the parameters. The first one of these gradients is useful to
obtain the minimum of the function, whereas the second one comes very
handy in the optimization of the parameters.
"""
import time
import numpy as np
from numpy.polynomial import chebyshev as cheby
from scipy.optimize import approx_fprime as grad_diff
import tensorly as tl
import tensorly.tenalg as tlalg

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
    with respect to the variables and parameters. A RMSE like
    cost function is also defined

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
    refdata : array
            Array of shape (N, f + 1) containing the reference
            geometries (points of the DVR) and corresponding energies.
            Required only if the gradient of the parameters is requested.
    """

    def __init__(self, chebdim, gdim, carray, refdata=None):
        self.chebdim = chebdim
        self.gdim = gdim
        self.carray = carray
        self.refdata = refdata

        # Total number of Chebyshev coefficients
        self.ncheb = np.sum(self.gdim) * self.chebdim

        # Creates the Chebyshev coefficient's tensor
        chebs_tk = np.array(
            np.split(carray[:self.ncheb],
                     carray[:self.ncheb].shape[0] / self.chebdim))
        self.chebs = np.array(
            np.split(chebs_tk, np.cumsum(self.gdim))[0:-1])

        # Creates the core tensor
        self.cora = carray[self.ncheb:].reshape(self.gdim)

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

    def rho(self):
        """
        Computes the value of the Root Mean Square cost function
        """
        dvrvals = self.refdata[:, :-1]
        e_ref = self.refdata[:, -1]
        e_sop = []
        for geo in dvrvals:
            e_sop.append(self.sop_fun(geo))
        e_sop = np.array(e_sop).flatten()
        return np.sqrt(np.mean((e_sop - e_ref) ** 2))

    @ArtDeco.timethis
    def sop_pargrad(self, q_array):
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

        Notes
        =====

        The derivatives of the Core tensor are fairly straightforward: they
        reduce to the Kronecker product of the SPPs. Chebyshev coefficients
        derivatives are a bit trickier. The key is that only the corresponding
        pure Chebyshev polynomial survives after differentiation.
        That is why the pure Chebyshev are computed in the main loop.
        """
        # Generates the matrices (or vectors) of the SPPs and pure Chebs
        v_matrices = []
        v_chebpure = []
        for kdof, m_kp in enumerate(self.gdim):
            v_ch = np.zeros(self.chebdim)
            coeff = [1]
            for miu in np.arange(self.chebdim):
                v_ch[miu] = cheby.chebval(q_array[kdof], coeff)
                coeff.insert(0, 0)

            v_kp = np.zeros(m_kp)
            for j_kp in np.arange(m_kp):
                v_kp[j_kp] = cheby.chebval(
                    q_array[kdof], self.chebs[kdof][j_kp])

            v_chebpure.append(v_ch)
            v_matrices.append(v_kp)

        v_chebpure = np.array(v_chebpure)
        v_matrices = np.array(v_matrices)

        # Core coefficients derivatives

        coreders = tlalg.kronecker(v_matrices)

        # Chebyshev coefficients derivatives

        chebders = np.zeros(self.ncheb)
        idxs = 0
        for kdof, m_kp in enumerate(self.gdim):
            for j_kp in np.arange(m_kp):
                matrices = np.copy(v_matrices)
                lonelycheb = np.zeros(m_kp)
                lonelycheb[j_kp] = v_chebpure[kdof, j_kp]
                matrices[kdof] = lonelycheb
                chebders[idxs] = tl.tucker_tensor.tucker_to_tensor(
                    (self.cora, matrices))
                idxs += 1

        # Concatenate results

        pargrad = np.concatenate((chebders, coreders))

        return pargrad


if __name__ == "__main__":
    CHEBDIM = 7
    GDIM = np.array([5, 5, 5, 5, 5, 5])
    CARRAY = np.loadtxt('params_init')
    QARR = np.array([2.6, 1.8, 2.2, 1.7, 1.9, np.pi])
    DATA = np.loadtxt('ref_ab')

    sop_hono = SopFbr(CHEBDIM, GDIM, CARRAY, DATA)

    print(sop_hono.sop_fun(QARR))
    print(sop_hono.sop_vargrad(QARR))
    print(sop_hono.sop_pargrad(QARR))
    T0 = time.time()
    grad_diff(QARR, sop_hono.sop_fun, 1e-7)
    TF = time.time()
    print(f"Numerical gradient in {TF - T0} s")
    print(grad_diff(QARR, sop_hono.sop_fun, 1e-7))

    print(sop_hono.rho())
