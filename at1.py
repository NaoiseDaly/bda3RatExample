from scipy.special import loggamma
from scipy.stats import uniform
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
    J = d.shape[0] # number of observations
    #first fraction is just raised to power of J
    first_fraction = J*(loggamma(alpha+beta)-loggamma(alpha)-loggamma(beta))
    second_fractions = [None for _ in range(J)]
    for j in range(J):
        top = loggamma(alpha+d.loc[j, "y.j"])+loggamma(beta+d.loc[j, "n.j"]-d.loc[j, "y.j"])
        bottom = loggamma(alpha+beta+d.loc[j, "n.j"])
        second_fractions[j] = top-bottom 
    
    return first_fraction+sum(second_fractions)

def unnormalised_posterior(alpha, beta, prior, data):
    mess = messy_part_in_posterior_log(alpha, beta, data)
    return np.exp(mess+prior(alpha, beta))



def normalise_grid(grid):
    return grid/grid.sum()

def marginal_of_alpha_on_grid(grid):
    "Careful: This depends on design of GRID "
    return grid.sum(axis =  0)

def beta_on_grid_given_alpha(grid, alpha):
    "Careful: This depends on design of GRID "
    alpha_i = np.where(np.isclose(grid["alpha"], alpha)) 
    alpha_i = alpha_i[0] #should be only one element in this
    beta_given_alpha_dist = grid["posterior"][alpha_i]/grid["posterior"][alpha_i].sum()
    
    return beta_given_alpha_dist

def sample_from_empirical_dist(n, values, dist,  random_state = None):
    """generates n samples from a collection values with distribution dist
    
    assumes the entry in values[i] occurs with probability dist[i]

    use random_state for reproducablity 
    """
    emp_cdf = np.cumsum(dist)
    draws = np.zeros(n)
    for i, u in enumerate(  uniform.rvs(size = n, random_state=random_state) ):
        draws[i] = values[np.min( np.where(emp_cdf > u) )]
    return  draws