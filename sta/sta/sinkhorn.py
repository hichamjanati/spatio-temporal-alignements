import warnings
import torch
import numpy as np


def wimg(p, q, K, epsilon=0.01, maxiter=2000, tol=1e-7, verbose=False,
         f0=0.):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    q: numpy array (width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    bold = torch.ones_like(p, requires_grad=False)
    b = bold.clone()
    Kb = K.mm(K.mm(b).t()).t()
    log = {'cstr': [], 'flag': 0}
    cstr = 10
    for i in range(maxiter):
        a = p / Kb
        Ka = K.t().mm(K.mm(a).t()).t()
        b = q / Ka
        Kb = K.mm(K.mm(b).t()).t()
        with torch.no_grad():
            cstr = abs(Kb * a - p).max().item()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    f = (torch.log(a + 1e-100) * p + torch.log(b + 1e-100) * q).sum() * epsilon
    f += f0
    return f, 0., Kb


def wbarycenter(P, K, epsilon=0.01, maxiter=2000, tol=1e-7, verbose=False,
                f0=0.):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein barycenter of P.

    """
    n_hists, width, _ = P.shape
    bold = torch.ones_like(P, requires_grad=False)
    b = bold.clone()
    Kb = convol_imgs(b, K)
    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    for i in range(maxiter):
        a = P / Kb
        Ka = convol_imgs(a, K.t())
        q = np.prod(Ka, dim=0) ** (1 / n_hists)
        Q = q[None, :, :]
        b = Q / Ka
        Kb = convol_imgs(b, K)
        with torch.no_grad():
            cstr = abs(a * Kb - P).max().item()
            log["cstr"].append(cstr)

            if cstr < tol and 0:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    f = (torch.log(a + 1e-100) * P + torch.log(b + 1e-100) * Q).sum()
    f *= epsilon
    f += f0
    return q, f


def wbarycenterkl(P, K, epsilon=0.01, gamma=1, maxiter=2000, tol=1e-7,
                  verbose=False, f0=0.):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein barycenter of P.

    """
    n_hists, width, _ = P.shape
    bold = torch.ones_like(P, requires_grad=False)
    b = bold.clone()
    Kb = convol_imgs(b, K)
    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    fi = gamma / (gamma + epsilon)
    for i in range(maxiter):
        a = (P / Kb) ** fi
        Ka = convol_imgs(a, K.t())
        q = ((Ka ** (1 - fi)).mean(dim=0))
        q = q ** (1 / (1 - fi))
        Q = q[None, :, :]
        b = (Q / Ka) ** fi
        Kb = convol_imgs(b, K)
        with torch.no_grad():
            cstr = abs(a * Kb - P).max().item()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = (a * Kb).sum()
    f = - (epsilon + 2 * gamma) * plsum + gamma * (P.sum())
    f += f0
    return q, f


def convol_imgs(imgs, K):
    kx = torch.einsum("...ij,kjl->kil", K, imgs)
    kxy = torch.einsum("...ij,klj->kli", K, kx)
    return kxy


def convol_old(imgs, K):
    kxy = torch.zeros_like(imgs)
    for i, img in enumerate(imgs):
        kxy[i] = K.mm(K.mm(img).t()).t()
    return kxy


def wimg_parallel(p, Q, K, epsilon=0.01, maxiter=2000, tol=1e-7,
                  verbose=False, f0=0.):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if Q.ndimension() == 2:
        Q = Q[None, :, :]
    bold = torch.ones_like(Q, requires_grad=False)
    b = bold.clone()
    Kb = convol_imgs(b, K)
    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    for i in range(maxiter):
        a = p / Kb
        Ka = convol_imgs(a, K.t())
        b = Q / Ka
        Kb = convol_imgs(b, K)
        with torch.no_grad():
            cstr = abs(Kb * a - p).mean().item()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    f = (torch.log(a + 1e-100) * p + torch.log(b + 1e-100) * Q).sum(dim=(1, 2))
    f *= epsilon
    f += f0

    return f, a, Kb


def wkl(p, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
        verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    Q = Q.t()
    if Q.ndimension() == 1:
        Q = Q[:, None]
    Qs = Q.sum(dim=0)
    psum = p.sum()
    f = gamma * (Qs + psum)
    idx = Qs > -1
    if compute_grad:
        g = gamma * (1 - 10 ** (20 * epsilon / gamma)) * torch.ones_like(Q)
    Q = Q[:, idx]
    if gamma == 0.:
        return wimg_parallel(p, Q, K, epsilon, maxiter, tol, verbose)
    if psum < -1:
        return f, g
    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    bold = torch.ones_like(Q, requires_grad=False)
    if Kb is None:
        Kb = K.mm(bold)
    for i in range(maxiter):
        a = (p[:, None] / Kb) ** fi
        Ka = K.t().mm(a)
        b = (Q / Ka) ** fi
        Kb = K.mm(b)
        with torch.no_grad():
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = (a * Kb).sum(0)
    f[idx] += epsilon * (f0 - plsum)
    f[idx] += - 2 * gamma * plsum
    # M = - epsilon * torch.log(K)
    # f = (convol_imgs(b, M * K) * a).sum((1, 2))
    if compute_grad:
        g = gamma * (1 - a ** (- epsilon / gamma))
        return f, g.t(), Kb
    return f, 0., Kb


def wkllog(p, q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
           verbose=False, f0=0., compute_grad=False):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if gamma == 0.:
        return wimg_parallel(p, q, K, epsilon, maxiter, tol, verbose)
    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    if compute_grad:
        g = (1 - 10 ** (20 * epsilon / gamma)) * torch.ones_like(p)
    ps = p.sum()
    qs = q.sum()
    if ps < -1 or qs < -1:
        return gamma * (qs + ps), g
    # if support:
    support_p = p > -1e-10
    support_q = q > -1e-10

    p = torch.log(p + 1e-15)
    q = torch.log(q + 1e-15)
    K = K - 1e-100
    p = p[support_p]
    q = q[support_q]

    K = K[support_p]
    K = K[:, support_q]
    b = torch.zeros_like(q, requires_grad=False)
    Kb = torch.logsumexp(K + b[None, :], dim=1)
    psumold = 0.
    for i in range(maxiter):
        a = fi * (p - Kb)
        Ka = torch.logsumexp(K.t() + a[None, :], dim=1)
        b = fi * (q - Ka)
        Kb = torch.logsumexp(K + b[None, :], dim=1)
        psum = torch.exp(a + Kb).sum().item()
        with torch.no_grad():
            cstr = abs(psumold - psum) / max(1, psumold, psum)
            psumold = psum
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    f = epsilon * (f0 - psum)
    f = f + gamma * (ps + qs - 2 * psum)
    # M = - epsilon * torch.log(K)
    # f = (convol_imgs(b, M * K) * a).sum((1, 2))
    if compute_grad:
        g = 1 - torch.exp(- epsilon * a / gamma)
        g *= gamma
        return f, g, Kb
    return f, 0, Kb


def negentropy_img(P, K, epsilon=0.01, gamma=1., maxiter=100, tol=1e-7,
                   verbose=False, f0=0., compute_grad=False, a=None):
    """Compute the Negentropy term W(p, p) elementwise.

    Parameters
    ----------
    P: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if P.ndimension() == 2:
        P = P[None, :, :]
    if gamma == 0.:
        fi = 1.
    else:
        fi = gamma / (gamma + epsilon)
    if a is None:
        a = torch.ones_like(P, requires_grad=False)
    aold = a.clone()
    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    # torch.set_grad_enabled(grad)
    for i in range(maxiter):
        Ka = convol_imgs(a, K)
        a = a ** 0.5 * (P / Ka) ** (fi / 2)

        with torch.no_grad():
            cstr = abs(a - aold).max() / max(1, a.max(), aold.max())
            aold = a.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    if gamma:
        psum = (a * Ka).sum((1, 2))
        f = - epsilon * psum
        f = f + gamma * (2 * P.sum((1, 2)) - 2 * psum)
    else:
        f = 2 * (P * torch.log(a + 1e-100)).sum((1, 2))
    f = f + epsilon * f0
    if compute_grad:
        grad = gamma * (1 - a ** (- epsilon / gamma))
        return f, grad, a
    return f, 0., a


def negentropy(P, K, epsilon=0.01, gamma=1., maxiter=100, tol=1e-7,
               verbose=False, f0=0., compute_grad=False, a=None):
    """Compute the Negentropy term W(p, p) elementwise.

    Parameters
    ----------
    P: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    P = P.t()
    if P.ndimension() == 1:
        P = P[:, None]
    if gamma == 0.:
        fi = 1.
    else:
        fi = gamma / (gamma + epsilon)
    aold = torch.ones_like(P, requires_grad=False)

    a = aold.clone()
    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    # torch.set_grad_enabled(grad)
    for i in range(maxiter):
        Ka = K.mm(a)
        a = a ** 0.5 * (P / Ka) ** (fi / 2)

        with torch.no_grad():
            cstr = abs(a - aold).max() / max(1, a.max(), aold.max())
            aold = a.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    if gamma:
        psum = (a * Ka).sum((0))
        f = - epsilon * psum
        f = f + gamma * (2 * P.sum(0) - 2 * psum)
    else:
        f = 2 * (P * torch.log(a + 1e-100)).sum(0)
    f = f + epsilon * f0
    if compute_grad:
        grad = gamma * (1 - a ** (- epsilon / gamma))
        return f, grad.t(), a
    return f, 0., a


def negentropy_log_(p, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                    verbose=False, f0=0., compute_grad=False, a=None):
    """Compute the Negentropy term W(p, p) elementwise.

    Parameters
    ----------
    P: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """

    if gamma == 0.:
        fi = 1.
    else:
        fi = gamma / (gamma + epsilon)
    if compute_grad:
        grad = (1 - 10 ** (20 * epsilon / gamma)) * torch.ones_like(p)
    ps = p.sum()
    # if support:
    support_p = p > -1e-10
    p = p[support_p]
    logp = torch.log(p + 1e-10)
    K = K - 1e-100
    grad = (1 - 10 ** (20 * epsilon / gamma)) * torch.ones_like(p)

    K = K[support_p][:, support_p]
    aold = torch.zeros_like(p, requires_grad=False)
    a = aold.clone()
    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    psumold = 0.
    for i in range(maxiter):
        ka = torch.logsumexp(K + a[None, :], dim=1)
        a = 0.5 * (a + fi * (logp - ka))
        psum = torch.exp(a + ka).sum().item()

        with torch.no_grad():
            cstr = abs(psumold - psum) / max(1, psumold, psum)
            psumold = psum
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    if gamma:
        f = - epsilon * psum
        f += gamma * (2 * ps - 2 * psum)
    else:
        f = 2 * (p * a).sum()
    f += epsilon * f0
    if compute_grad:
        grad[support_p] = 1 - torch.exp(- epsilon * a / gamma)
        grad *= gamma
        return f, grad, a
    return f, 0., a


def negentropy_log(P, K, epsilon=0.01, gamma=1., maxiter=100, tol=1e-7,
                   verbose=False, f0=0., compute_grad=False, a=None):
    if P.ndimension() == 1:
        P = P[None, :]
    n_times, dimension = P.shape
    f = torch.zeros(n_times, dtype=P.dtype, device=P.device)
    grad = torch.zeros_like(P)
    for i, p in enumerate(P):
        ff, gg = negentropy_log_(p, K, epsilon, gamma, maxiter, tol,
                                 verbose, f0, compute_grad=compute_grad)
        f[i] = ff
        grad[i] = gg
    if compute_grad:
        return f, grad, a
    return f, 0., a


def wimgkl(p, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
           verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if Q.ndimension() == 2:
        Q = Q[None, :, :]
    Qs = Q.sum(dim=(1, 2))
    psum = p.sum()
    f = gamma * (Qs + psum)

    if gamma == 0.:
        return wimg_parallel(p, Q, K, epsilon, maxiter, tol, verbose)
    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    b = torch.ones_like(Q, requires_grad=False)
    if Kb is None:
        Kb = convol_imgs(b, K)
    bold = b.clone()
    for i in range(maxiter):
        a = (p / Kb) ** fi
        Ka = convol_imgs(a, K.t())
        b = (Q / Ka) ** fi
        Kb = convol_imgs(b, K)
        with torch.no_grad():
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break

            if cstr < tol or torch.isnan(psum).any():
                break
    if i == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = (a * Kb).sum((1, 2))

    f += epsilon * (f0 - plsum)
    f += - 2 * gamma * plsum
    if compute_grad:
        g = gamma * (1 - a ** (- epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def convol_huge_imgs(imgs, K):
    n, m, dimension, dimension = imgs.shape
    out = convol_imgs(imgs.reshape(n * m, dimension, dimension), K)
    out = out.reshape(n, m, dimension, dimension)
    return out


def convol_huge(imgs, K):
    dimension, n, m = imgs.shape
    out = K.mm(imgs.reshape(dimension, n * m))
    out = out.reshape(dimension, n, m)
    return out


def convol_huge_log(imgs, C):
    dimension, n, m = imgs.shape
    imgs = imgs.reshape(dimension, n * m)
    out = torch.logsumexp(C[:, :, None] + imgs[None, :], dim=1)
    out = out.reshape(dimension, n, m)
    return out


def convol_imgs_log(imgs, C):
    """Compute log separable kernal application."""
    n, dimension, dimension = imgs.shape
    x = (torch.logsumexp(C[None, None, :, :] + imgs[:, :, None], dim=-1))
    x = torch.logsumexp(C.t()[None, :, :, None] + x[:, :, None], dim=1)
    return x.reshape(n, dimension, dimension)


def convol_huge_imgs_log(imgs, C):
    """Compute log separable kernal application."""
    n, m, dimension, dimension = imgs.shape
    imgs = imgs.reshape(n * m, dimension, dimension)
    x = (torch.logsumexp(C[None, None, :, :] + imgs[:, :, None], dim=-1))
    x = torch.logsumexp(C.t()[None, :, :, None] + x[:, :, None], dim=1)
    return x.reshape(n, m, dimension, dimension)


def monster_img(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-8,
                verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if Q.ndimension() == 2:
        Q = Q[None, :, :]
    if P.ndimension() == 2:
        P = P[None, :, :]
    Qs = Q.sum(dim=(1, 2))
    Ps = P.sum(dim=(1, 2))

    f = gamma * (Qs[None, :] + Ps[:, None]) + epsilon * f0
    idq = Qs > -1e-2
    idp = Ps > -1e-2
    if compute_grad:
        g = torch.ones(len(P), *Q.shape, dtype=Q.dtype, device=Q.device)
        g *= gamma * (1 - 10 ** (20 * epsilon / gamma))
    Q = Q[idq]
    P = P[idp]

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    if Kb is None:
        b = torch.ones_like(Q)[None, :]
        Kb = convol_huge_imgs(b, K)

    bold = b.clone()
    for i in range(maxiter):
        a = (P[:, None, :, :] / Kb) ** fi
        Ka = convol_huge_imgs(a, K.t())
        b = (Q[None, :, :, :] / Ka) ** fi
        Kb = convol_huge_imgs(b, K)

        with torch.no_grad():
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
            if torch.isnan(Kb).any():
                warnings.warn("Numerical Errors ! Stopped at last stable "
                              "iteration.")
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = (a * Kb).sum((2, 3))

    for j in range(len(Q)):
        if idq[j]:
            f[idp, j] += - (epsilon + 2 * gamma) * plsum[:, j]

    if compute_grad:
        for j in range(len(Q)):
            if idq[j]:
                g[idp, j] = gamma * (1 - a[:, j] ** (- epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def monster_img_log(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                    verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if Q.ndimension() == 2:
        Q = Q[None, :, :]
    if P.ndimension() == 2:
        P = P[None, :, :]
    Qs = Q.sum(dim=(1, 2))
    Ps = P.sum(dim=(1, 2))
    f = gamma * (Qs[None, :] + Ps[:, None]) + epsilon * f0
    P = torch.log(P + 1e-100)
    Q = torch.log(Q + 1e-100)

    if compute_grad:
        g = torch.ones(len(P), *Q.shape, dtype=Q.dtype, device=Q.device)

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    b = torch.zeros(len(Ps), *Q.shape, dtype=Q.dtype, device=Q.device)
    Kb = convol_huge_imgs_log(b, K)
    bold = b.clone()
    for i in range(maxiter):
        a = fi * (P[:, None, :, :] - Kb)
        Ka = convol_huge_imgs_log(a, K.t())
        b = fi * (Q[None, :, :, :] - Ka)
        Kb = convol_huge_imgs_log(b, K)
        with torch.no_grad():
            if torch.isnan(Kb).any():
                raise ValueError("Nan values found in Sinkhorn :(")
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3

    f += - (epsilon + 2 * gamma) * torch.exp(a + Kb).sum((2, 3))

    if compute_grad:
        g = gamma * (1 - torch.exp(- a * epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def monster(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
            verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if compute_grad:
        g = torch.ones(len(P), *Q.shape, dtype=Q.dtype, device=Q.device)
        g *= gamma * (1 - 10 ** (20 * epsilon / gamma))
    Q = Q.t()
    P = P.t()
    if Q.ndimension() == 1:
        Q = Q[:, None]
    if P.ndimension() == 1:
        P = P[:, None]
    Qs = Q.sum(dim=0)
    Ps = P.sum(dim=0)
    f = gamma * (Qs[None, :] + Ps[:, None]) + epsilon * f0
    n, m = f.shape
    idq = Qs > -1e-2
    idp = Ps > -1e-2

    Q = Q[:, idq]
    P = P[:, idp]

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    if Kb is None:
        b = torch.ones((*P.shape, Q.shape[1]), dtype=P.dtype, device=P.device)
        Kb = convol_huge(b, K)

    bold = b.clone()
    for i in range(maxiter):
        a = (P[:, :, None] / Kb) ** fi
        Ka = convol_huge(a, K.t())
        b = (Q[:, None, :] / Ka) ** fi
        Kb = convol_huge(b, K)

        with torch.no_grad():
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
            if torch.isnan(Kb).any():
                warnings.warn("Numerical Errors ! Stopped at last stable "
                              "iteration.")
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = (a * Kb).sum(0)

    for j in range(m):
        if idq[j]:
            f[idp, j] += - (epsilon + 2 * gamma) * plsum[:, j]
    if compute_grad:
        for j in range(m):
            if idq[j]:
                g[idp, j] = gamma * (1 - a[:, :, j].t() ** (- epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def monster_log(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if compute_grad:
        g = torch.ones(len(P), *Q.shape, dtype=Q.dtype, device=Q.device)
        g *= gamma * (1 - 10 ** (20 * epsilon / gamma))
    Q = Q.t()
    P = P.t()
    if Q.ndimension() == 1:
        Q = Q[:, None]
    if P.ndimension() == 1:
        P = P[:, None]
    Qs = Q.sum(dim=0)
    Ps = P.sum(dim=0)
    f = gamma * (Qs[None, :] + Ps[:, None]) + epsilon * f0
    n, m = f.shape
    idq = Qs > -1e-2
    idp = Ps > -1e-2

    Q = Q[:, idp]
    P = P[:, idq]

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    b = torch.zeros_like((*P.shape, Q.shape[1]), requires_grad=False)

    if Kb is None:
        Kb = convol_huge_log(b, K)

    bold = b.clone()
    for i in range(maxiter):
        a = fi * (P[:, :, None] - Kb)
        Ka = convol_huge_log(a, K.t())
        b = fi * (Q[:, None, :] - Ka)
        Kb = convol_huge_log(b, K)

        with torch.no_grad():
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break

    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = np.exp(a + Kb).sum(0)

    for j in range(m):
        if idq[j]:
            f[idp, j] += - (epsilon + 2 * gamma) * plsum[:, j]
    if compute_grad:
        for j in range(m):
            if idq[j]:
                g[idp, j] = 1 - torch.exp(- epsilon * a[:, :, j] / gamma)
                g[idp, j] *= gamma
        return f, g, Kb
    return f, 0., Kb


def divergencekl(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                 verbose=False, f0=0., compute_grad=False, log=False,
                 Kb=None, axx=None, ayy=None):
    if log:
        if P.ndimension() == 3:
            wxy = monster_img_log
            wxx = negentropy_img_log
        else:
            wxy = monster_log
            wxx = negentropy_log
    else:
        if P.ndimension() == 3:
            wxy = monster_img
            wxx = negentropy_img
        else:
            wxy = monster
            wxx = negentropy
    fxy, gxy, Kb = wxy(P, Q, K, epsilon, gamma, maxiter, tol,
                       verbose, f0, compute_grad, Kb=Kb)
    fxx, gxx, axx = wxx(P, K, epsilon, gamma, maxiter, tol,
                        verbose, f0, compute_grad, a=axx)
    fyy, _, ayy = wxx(Q, K, epsilon, gamma, maxiter, tol,
                      verbose, f0, compute_grad=False, a=ayy)
    f = fxy - 0.5 * (fxx[:, None] + fyy[None, :])
    G = 0.
    if compute_grad:
        G = gxy - gxx[:, None]
    # del fxy, gxy, fxx, gxx, fyy
    return f, G, Kb, axx, ayy


def negentropy_img_log(P, K, epsilon=0.01, gamma=1., maxiter=100, tol=1e-7,
                       verbose=False, f0=0., compute_grad=False, a=None):
    """Compute the Negentropy term W(p, p) elementwise.

    Parameters
    ----------
    P: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if P.ndimension() == 2:
        P = P[None, :, :]
    if gamma == 0.:
        fi = 1.
    else:
        fi = gamma / (gamma + epsilon)
    Ps = P.sum((1, 2))
    P = torch.log(P + 1e-10)
    if a is None:
        a = torch.zeros_like(P, requires_grad=False)
    aold = a.clone()
    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    # torch.set_grad_enabled(grad)
    for i in range(maxiter):
        Ka = convol_imgs_log(a, K)
        a = 0.5 * (a + fi * (P - Ka))
        with torch.no_grad():
            cstr = abs(a - aold).max() / max(1, a.max(), aold.max())
            aold = a.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    if gamma:
        psum = torch.exp(a + Ka).sum((1, 2))
        f = - epsilon * psum
        f = f + gamma * (2 * Ps - 2 * psum)
    else:
        f = 2 * (P * a).sum((1, 2))
    f = f + epsilon * f0
    if compute_grad:
        grad = gamma * (1 - torch.exp(- a * epsilon / gamma))
        return f, grad, a
    return f, 0., a


def wimgkl_parallel(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                    verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence (p1, q1), (p2, q2) ...

    Parameters
    ----------
    P: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if Q.ndimension() == 2:
        Q = Q[None, :, :]
    Qs = Q.sum(dim=(1, 2))
    Ps = P.sum(dim=(1, 2))
    f = gamma * (Qs + Ps)

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    bold = torch.ones_like(Q, requires_grad=False)
    if Kb is None:
        b = bold.clone()
        Kb = convol_imgs(b, K)
    plsumold = torch.zeros(len(Q), dtype=Q.dtype, device=Q.device)
    for i in range(maxiter):
        a = (P / Kb) ** fi
        Ka = convol_imgs(a, K.t())
        b = (Q / Ka) ** fi
        Kb = convol_imgs(b, K)
        plsum = (a * Kb).sum((1, 2))
        with torch.no_grad():
            cstr = abs(plsumold - plsum).max() / max(1, plsumold.max(),
                                                     plsum.max())
            plsumold = plsum
            log["cstr"].append(cstr)

            if cstr < tol or torch.isnan(plsum).any():
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    f += epsilon * (f0 - plsum)
    f += - 2 * gamma * plsum
    if compute_grad:
        g = gamma * (1 - a ** (- epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def wkl_parallel(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                 verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence (p1, q1), (p2, q2) ...

    Parameters
    ----------
    P: numpy array (n_hists, dimension)
        Must be non-negative.
    Q: numpy array (n_hists, dimension)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    Q = Q.t()
    P = P.t()
    if Q.ndimension() == 1:
        Q = Q[:, None]
    if P.ndimension() == 1:
        P = P[:, None]

    Qs = Q.sum(dim=0)
    Ps = P.sum(dim=0)
    f = gamma * (Qs + Ps)

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    bold = torch.ones_like(Q, requires_grad=False)
    if Kb is None:
        b = bold.clone()
        Kb = K.mm(b)
    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    bold = torch.ones_like(Q, requires_grad=False)
    if Kb is None:
        Kb = K.mm(bold)

    for i in range(maxiter):
        a = (P / Kb) ** fi
        Ka = K.t().mm(a)
        b = (Q / Ka) ** fi
        Kb = K.mm(b)
        with torch.no_grad():
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = (a * Kb).sum(0)
    f += epsilon * (f0 - plsum)
    f += - 2 * gamma * plsum
    if compute_grad:
        g = gamma * (1 - a ** (- epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def amarikl(P, Q, Qtild, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
            verbose=False, f0=0., compute_grad=False, log=False,
            Kb=None, normalize=True):
    if log:
        if P.ndimension() == 3:
            wxy = monster_img_log
            wsym = wimgkl_parallel
        else:
            wxy = monster
            wsym = wimgkl_parallel
    else:
        if P.ndimension() == 3:
            wxy = monster_img
            wsym = wimgkl_parallel
        else:
            wxy = monster
            wsym = wkl_parallel
    fxy, gxy, Kb = wxy(P, Qtild, K, epsilon, gamma, maxiter, tol,
                       verbose, f0, compute_grad, Kb=Kb)
    if normalize:
        fyky, _, _ = wsym(Q, Qtild, K, epsilon, gamma, maxiter, tol,
                          verbose, f0, compute_grad)
    else:
        fyky = torch.zeros(len(Q), dtype=P.dtype)
    G = 0.
    if compute_grad:
        G = gxy
    # del fxy, gxy, fxx, gxx, fyy
    return fxy, G, Kb, fyky
