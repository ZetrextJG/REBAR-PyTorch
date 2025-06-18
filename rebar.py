import torch


def g(u, theta, eps = 1e-8):
    """ 
    Differentiable reparametrization of z.

    Givem u ~ U(0, 1), and theta from (0, 1).
    Return a sample from p(z). Logistic distribution.
    """
    return torch.log(theta / (1 - theta) + eps) + torch.log(u / (1 - u) + eps)


def g_tilde(v, b, theta, eps = 1e-8):
    """
    Differentiable reparametrization of z|b.

    Given v ~ U(0, 1), b ~ Bernoulli(theta).
    and theta from (0, 1).

    Return a sample from the distribution p(z|b).
    """
    zb1 = torch.log((v / (1 - v) + eps) * (1/(1-theta)) + 1)
    zb0 = -torch.log((v / (1 - v) + eps) * (1/theta) + 1)
    return b * zb1 + (b-1) * zb0


def H(z):
    """
    The hard threshold function.
    """
    return torch.where(z >= 0, 1, 0)


def sigma(z, lmbda = 1, eps= 1e-8):
    """
    Sigmoid with a temperature parameter.
    """
    return torch.sigmoid(z / (lmbda + eps)) 


def g_lmbda(u, theta, lmbda = 1, eps= 1e-8):
    """
    Differentiable reparametrization of z_lmbda.

    After sigma this makes an alternative to sigma(g(u, theta), lmbda).
    lmbda must be the same for both g_lmbda and sigma.
    """
    quadratic = (lmbda**2 + lmbda + 1)/(lmbda + 1)
    z_lmbda = quadratic * torch.log(theta / (1 - theta) + eps) + torch.log(u / (1 - u) + eps)
    return z_lmbda


def log_likelihood_bern(b, theta, eps=1e-6):
    """
    Log likelihood of Bernoulli distribution.
    
    b: tensor of binary values (0 or 1)
    theta: probability of success (in (0, 1))
    """
    return b * torch.log(theta + eps) + (1 - b) * torch.log(1 - theta + eps)