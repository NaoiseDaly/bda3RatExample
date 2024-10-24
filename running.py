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

GRID["posterior"] = output 
print(f"before {GRID["posterior"].sum()}, {GRID["posterior"].shape}")
GRID["posterior"] = normalise_grid(GRID["posterior"])
print(f"after {GRID["posterior"].sum()}, {GRID["posterior"].shape}")

plt.contourf(
        GRID["alpha"], GRID["beta"], GRID["posterior"]
    )
plt.xlabel("alpha")
plt.ylabel("beta")
plt.colorbar()
plt.show()



plt.contourf(
        GRID["logit_mean"], GRID["log_sample_size"], GRID["posterior"]
    )
plt.xlabel("logit_mean")
plt.ylabel("log_sample_size")
plt.colorbar()
plt.show()

posterior_alpha = marginal_of_alpha_on_grid(GRID["posterior"])
plt.bar(GRID["alpha"], posterior_alpha)
print(sum(posterior_alpha))
plt.show()


