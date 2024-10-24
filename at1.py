from scipy.special import loggamma
import numpy as np


def transform_to_alpha_beta(logit_mean, log_sample_size):
    """Helper function"""
    beta = np.exp(log_sample_size)/(1+np.exp(logit_mean))
    alpha = np.exp(logit_mean)*beta
    return alpha, beta

def transform_from_alpha_beta(alpha, beta):
    """Helper function"""
    logit_mean = np.log(alpha/beta)
    log_sample_size = np.log(alpha+beta)
    return logit_mean, log_sample_size



def some_diffuse_prior_log(alpha, beta):
    """The log of a suggested prior."""
    return np.log(alpha+beta)*(-5/2)

def uniform_prior_log(alpha, beta):
    return 0

def messy_part_in_posterior_log(alpha, beta, d):
    """The log of the 'non-prior' part of the posterior"""
    parts = [None for _ in range(d.shape[0])]
    for j in range(d.shape[0]):
        top = loggamma(alpha+beta)+loggamma(alpha+d.loc[j, "y.j"])+loggamma(beta+d.loc[j, "n.j"]-d.loc[j, "y.j"])
        bottom = loggamma(alpha)+loggamma(beta)+loggamma(alpha+beta+d.loc[j, "n.j"])
        parts[j] = top-bottom 
    
    return sum(parts)

def unnormalised_posterior(alpha, beta, prior, data):
    mess = messy_part_in_posterior_log(alpha, beta, data)
    return np.exp(mess+prior(alpha, beta))



def normalise_grid(grid):
    return grid/grid.sum()

def marginal_of_alpha_on_grid(grid):
    "Careful: This depends on design of GRID "
    return grid.sum(axis =  0)
