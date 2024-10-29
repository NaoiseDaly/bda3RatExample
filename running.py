from at1 import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform

dataset = pd.read_csv("tumour-data.txt", names = ("y.j", "n.j"))

SIZE_OF_GRID = 100
#equal spacing on the transformed scale does not mean equal spacing on the original scale
alpha_boundaries, beta_boundaries = transform_to_alpha_beta((-2.3,-1.3), (1, 5))
STEP_SIZE_IN_ALPHA = (alpha_boundaries[1]-alpha_boundaries[0])/SIZE_OF_GRID
STEP_SIZE_IN_BETA = (beta_boundaries[1]-beta_boundaries[0])/SIZE_OF_GRID

GRID ={
    "alpha": np.linspace(*alpha_boundaries, num = SIZE_OF_GRID),
    "beta" : np.linspace(*beta_boundaries, num = SIZE_OF_GRID)
}
GRID["logit_mean"], GRID["log_sample_size"] = transform_from_alpha_beta(GRID["alpha"], GRID["beta"])

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

s = 2000 #number of simulated samples 
posterior_alpha = marginal_of_alpha_on_grid(GRID["posterior"])
draws_of_alpha = sample_from_empirical_dist(s, GRID["alpha"], posterior_alpha)

draws_of_beta_given_alpha = np.array([None for _ in range(s)], dtype = float)

for i, alpha_sample in enumerate(draws_of_alpha):
    beta_dist = beta_on_grid_given_alpha(GRID, alpha_sample)
    beta_sample = sample_from_empirical_dist(1, GRID["beta"], beta_dist )[0]
    draws_of_beta_given_alpha[i] = beta_sample

#add jitter to sampled pairs
draws_of_alpha += uniform.rvs(size = s, scale = STEP_SIZE_IN_ALPHA)
draws_of_beta_given_alpha += uniform.rvs(size = s, scale = STEP_SIZE_IN_BETA)


fig, (ax1, ax2) = plt.subplots(1,2)

ax1.scatter(draws_of_alpha, draws_of_beta_given_alpha)
ax1.set_xlabel("alpha")
ax1.set_ylabel("beta")

ax2.scatter(*transform_from_alpha_beta(draws_of_alpha, draws_of_beta_given_alpha))
ax2.set_xlabel("logit_mean")
ax2.set_ylabel("log_sample_size")
plt.show()

fig, (ax1, ax2) = plt.subplots(1,2, sharex = True)
ax1.bar(GRID["alpha"], posterior_alpha)
ax1.set_title("marginal unnormalised posterior of alpha")
values, heights = np.unique(draws_of_alpha, return_counts=True)
ax2.bar(values, heights)
ax2.set_title(f"frequency of {s} simulated draws from posterior")
plt.show()