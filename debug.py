from at1 import *
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("tumour-data.txt", names = ("y.j", "n.j"))

alpha_on_grid, beta_on_grid = transform_to_alpha_beta(GRID["logit_mean"], GRID["log_sample_size"])
alpha_on_grid_mesh, beta_on_grid_mesh = np.meshgrid(alpha_on_grid, beta_on_grid)


output = unnormalised_posterior(
    alpha_on_grid_mesh,beta_on_grid_mesh
    ,some_diffuse_prior_log,dataset)

GRID["posterior"] = output 

plt.contourf(
        GRID["logit_mean"], GRID["log_sample_size"], GRID["posterior"]
    )

plt.colorbar()
plt.show()
# print(dataset.loc[0])
