import torch
from scipy.linalg import toeplitz

#from sdtw.distance import SquaredEuclidean
from sdtw_div.numba_ops import squared_euclidean_cost
import gc


from .sinkhorn import divergencekl, amarikl, convol_imgs
from .utils import tonumpy


class SinkhornDistance(object):
    def __init__(self, x, y, K, return_grad=True,
                 ot=True, autodiff=False, warmstart=True,
                 amari=None, **ot_params):
        """
        Parameters
        ----------
        x: array, shape = [m, d]
            First time series.
        y: array, shape = [n, d]
            Second time series.
        """
        self.x = x
        self.y = y
        self.K = K
        self.ot_params = ot_params
        self.autodiff = autodiff
        self.return_grad = return_grad
        self.ot = ot
        self.warmstart = warmstart
        self.Kb, self.axx, self.ayy, self.fyky = None, None, None, None
        self.amari = amari
        self._multiple = False

    def update_x_y(self, x, y):
        """Set the parameters"""
        self.x = x
        if self.y != y:
            self.fyky = None
        self.y = y

    def compute(self):
        """
        Compute distance matrix.
        Returns
        -------
        D: array, shape = [m, n]
            Distance matrix.
        """
        if self.y.ndimension() > self.x.ndimension():
            n_time_series, length_y = self.y.shape[:2]
            self._multiple = True
            self.y = self.y.reshape(n_time_series * length_y,
                                    *self.y.shape[2:])
        else:
            self._multiple = False
            n_time_series, length_y = 1, len(self.y)

        if not self.ot:
            n = len(self.x)
            m = len(self.y)
            D = squared_euclidean_cost(tonumpy(self.x).reshape(n, -1),
                                 tonumpy(self.y).reshape(m, -1))
            return D#.compute()
        # if self.dirac:
        #     self.x = self.x.reshape(m, -1)
        #     self.y = self.y.reshape(n, -1)
        #     ix = torch.argmax(self.x, dim=-1)
        #     iy = torch.argmax(self.y, dim=-1)
        #     w = self.K[ix][:, iy]
        #     return tonumpy(w)
        if self.amari is not None:
            Ksum = self.amari ** self.ot_params["gamma"]
            if Ksum.ndimension() == 2:
                self.ytild = convol_imgs(self.y / Ksum[None, :], self.K)
            else:
                self.ytild = (self.K / Ksum[:, None]).mm(self.y.t()).t()
            normalize = self.fyky is None
            ww, G, Kb, fyky = amarikl(self.x, self.y, self.ytild, self.K,
                                      normalize=normalize,
                                      compute_grad=self.return_grad,
                                      Kb=self.fyky,
                                      **self.ot_params)
            ww = ww - fyky[None, :]
            if self.warmstart:
                self.Kb = Kb
            self.fyky = fyky
        else:
            ww, G, Kb, axx, ayy = divergencekl(self.x, self.y, self.K,
                                               compute_grad=self.return_grad,
                                               Kb=self.Kb, axx=self.axx,
                                               **self.ot_params)

        ww = ww.reshape(len(ww), n_time_series, length_y)
        if self.return_grad:
            G = G.reshape(len(ww), n_time_series, length_y, *G.shape[2:])
            self.jac = torch.zeros_like(G)
            if self.autodiff:
                for i, wx in enumerate(ww):
                    for k, wxy in enumerate(wx):
                        for j, wxy_k in enumerate(wxy):
                            wxy_k.backward(retain_graph=True)
                            self.jac[i, k, j] = self.x.grad[i]
                            self.x.grad.zero_()
            else:
                if not self._multiple:
                    G = G.squeeze()
                self.jac = G
        w = tonumpy(ww)
        if not self._multiple:
            w = w.squeeze()

        del ww
        gc.collect()

        self._dist_matrix = w
        torch.cuda.empty_cache()

        return w
