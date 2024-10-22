import pandas as pd
from scipy.special import loggamma
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("tumour-data.txt", names = ("y.j", "n.j"))


def some_diffuse_prior_log(alpha, beta):
    return np.log(alpha+beta)*(-5/2)

def messy_part_in_posterior_log(alpha, beta, d):
    parts = [None for _ in range(d.shape[0])]
    for j in range(d.shape[0]):
        top = loggamma(alpha+beta)+loggamma(alpha+d.loc[j, "y.j"])+loggamma(beta+d.loc[j, "n.j"]-d.loc[j, "y.j"])
        bottom = loggamma(alpha)*loggamma(beta)*loggamma(alpha+beta+d.loc[j, "n.j"])
        parts[j] = top-bottom #log scale
    
    return sum(parts)

def unnormalised_posterior(alpha, beta, prior, data):
    mess = messy_part_in_posterior_log(alpha, beta, data)
    print(mess)
    print(np.exp(mess))
    # return prior(alpha, beta)*
    return mess+prior(alpha, beta)






def transform_to_alpha_beta(logit_mean, log_sample_size):
    beta = np.exp(log_sample_size)/(1+np.exp(logit_mean))
    alpha = np.exp(logit_mean)*beta
    return alpha, beta

def transform_from_alpha_beta(alpha, beta):
    logit_mean = np.log(alpha/beta)
    log_sample_size = np.log(alpha+beta)
    return logit_mean, log_sample_size



GRID = {"logit_mean":np.linspace(-2.3,-1.3, num = 100)
     ,"log_sample_size":np.linspace(1,5, num = 100)
     ,"posterior": np.zeros((100,2))
     }   

# alpha_on_grid, beta_on_grid 
# alpha_on_grid_mesh, beta_on_grid_mesh = np.meshgrid(alpha_on_grid, beta_on_grid)#make them into co-ord matrices
# posterior_on_grid = unnormalised_posterior(alpha_on_grid,beta_on_grid
#                                            ,some_diffuse_prior, dataset)

