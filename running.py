from at1 import *
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("tumour-data.txt", names = ("y.j", "n.j"))


GRID = {"logit_mean":np.linspace(-2.3,-1.3, num = 1000)
     ,"log_sample_size":np.linspace(1,5, num = 1000)
     } 

alpha_on_grid, beta_on_grid = transform_to_alpha_beta(GRID["logit_mean"], GRID["log_sample_size"])
alpha_on_grid_mesh, beta_on_grid_mesh = np.meshgrid(alpha_on_grid, beta_on_grid, sparse = True)


output = unnormalised_posterior(
    alpha_on_grid_mesh,beta_on_grid_mesh
    ,some_diffuse_prior_log,dataset)

GRID["posterior"] = output 

plt.contourf(
        alpha_on_grid, beta_on_grid, GRID["posterior"]
    )

plt.colorbar()
plt.show()


output2 = unnormalised_posterior(
    alpha_on_grid_mesh,beta_on_grid_mesh
    ,uniform_prior_log,dataset)

plt.contourf(
        alpha_on_grid, beta_on_grid, output2
    )

plt.colorbar()
plt.show()
# print(dataset.loc[0])
