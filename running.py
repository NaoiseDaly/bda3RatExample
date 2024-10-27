from at1 import *
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("tumour-data.txt", names = ("y.j", "n.j"))


GRID = {"logit_mean":np.linspace(-2.3,-1.3, num = 100)
     ,"log_sample_size":np.linspace(1,5, num = 100)
     } 
GRID["alpha"], GRID["beta"] = transform_to_alpha_beta(GRID["logit_mean"], GRID["log_sample_size"])

alpha_on_grid_mesh, beta_on_grid_mesh = np.meshgrid(GRID["alpha"], GRID["beta"], sparse = True)


output = unnormalised_posterior(
    alpha_on_grid_mesh,beta_on_grid_mesh
    ,some_diffuse_prior_log,dataset)

GRID["posterior"] = normalise_grid(output)



fig, (ax1, ax2) = plt.subplots(1,2)
plot1 = ax1.contourf(
        GRID["alpha"], GRID["beta"], GRID["posterior"]
    )
ax1.set_xlabel("alpha")
ax1.set_ylabel("beta")
plt.colorbar(plot1, ax =ax1)

plot2 = ax2.contourf(
        GRID["logit_mean"], GRID["log_sample_size"], GRID["posterior"]
    )
ax2.set_xlabel("logit_mean")
ax2.set_ylabel("log_sample_size")
plt.colorbar(plot2, ax =ax2)
plt.show()

s = 20000 #number of simulated samples 
posterior_alpha = marginal_of_alpha_on_grid(GRID["posterior"])
draws_of_alpha = sample_from_empirical_dist(s, GRID["alpha"], posterior_alpha)

fig, (ax1, ax2) = plt.subplots(1,2, sharex = True)
ax1.bar(GRID["alpha"], posterior_alpha)
ax1.set_title("marginal unnormalised posterior of alpha")
values, heights = np.unique(draws_of_alpha, return_counts=True)
ax2.bar(values, heights)
ax2.set_title(f"frequency of {s} simulated draws from posterior")
plt.show()



