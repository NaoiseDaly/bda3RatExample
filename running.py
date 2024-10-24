from at1 import *
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("tumour-data.txt", names = ("y.j", "n.j"))


GRID = {"logit_mean":np.linspace(-2.3,-1.3, num = 1000)
     ,"log_sample_size":np.linspace(1,5, num = 1000)
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
# cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(plot1, ax =ax1)



plot2 = ax2.contourf(
        GRID["logit_mean"], GRID["log_sample_size"], GRID["posterior"]
    )
ax2.set_xlabel("logit_mean")
ax2.set_ylabel("log_sample_size")
plt.colorbar(plot2, ax =ax2)
plt.show()

# posterior_alpha = marginal_of_alpha_on_grid(GRID["posterior"])
# plt.bar(GRID["alpha"], posterior_alpha)
# print(sum(posterior_alpha))
# plt.show()


