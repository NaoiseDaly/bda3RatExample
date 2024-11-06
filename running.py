from at1 import *
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("tumour-data.txt", names = ("y.j", "n.j"))

SIZE_OF_GRID = 100+1
GRID = {}
GRID["logit_mean"] = np.linspace(-2.3,-1.3, num = SIZE_OF_GRID)
GRID["log_sample_size"] = np.linspace(1,5, num = SIZE_OF_GRID)
GRID["alpha"], GRID["beta"] = transform_to_alpha_beta(GRID["logit_mean"], GRID["log_sample_size"])

#it has to be done in this order
logit_mean_mesh, log_sample_size_mesh = np.meshgrid(GRID["logit_mean"], GRID["log_sample_size"],sparse = True)
alpha_on_grid_mesh, beta_on_grid_mesh = transform_to_alpha_beta(
    logit_mean_mesh, log_sample_size_mesh
    )

GRID["log_posterior_original"] = log_unnormalised_posterior_original(
    alpha_on_grid_mesh,beta_on_grid_mesh
    ,some_diffuse_prior_log,dataset
)
GRID["log_posterior_transform"] = log_unnormalised_posterior_transform(
    alpha_on_grid_mesh,beta_on_grid_mesh
    ,some_diffuse_prior_log,dataset
)


fig, (ax1, ax2) = plt.subplots(1,2)
plot1 = ax1.contourf(
        GRID["alpha"], GRID["beta"], GRID["log_posterior_original"]
        ,levels =np.linspace(
            GRID["log_posterior_original"].min(),GRID["log_posterior_original"].max(), 30 )
    )
ax1.set_title("log unnormalised posterior on alpha, beta")
ax1.set_xlabel("alpha")
ax1.set_ylabel("beta")
plt.colorbar(plot1, ax =ax1)

plot2 = ax2.contourf(
        GRID["logit_mean"], GRID["log_sample_size"], GRID["log_posterior_transform"]
        ,levels =np.linspace(
            GRID["log_posterior_transform"].min(),GRID["log_posterior_transform"].max(), 30 ) 
    )
ax2.set_title("log unnormalised posterior on logit mean and log sample size")
ax2.set_xlabel("logit_mean")
ax2.set_ylabel("log_sample_size")
plt.colorbar(plot2, ax =ax2)
plt.show()

#the plot has better contour lines before doing this
# subtract the maximum before exponentiating for numeric stability 
GRID["log_posterior_transform"] -= GRID["log_posterior_transform"].max()
GRID["log_posterior_original"] -= GRID["log_posterior_original"].max()
#exponentiate to get unnormalised probabilities
GRID["posterior_transform"] = np.exp(GRID["log_posterior_transform"])
GRID["posterior_original"] = np.exp(GRID["log_posterior_original"])
#normalise
GRID["posterior_transform"] /= GRID["posterior_transform"].sum()
GRID["posterior_original"] /= GRID["posterior_original"].sum()



fig, (ax1, ax2) = plt.subplots(1,2)
plot1 = ax1.contourf(
        GRID["alpha"], GRID["beta"], GRID["posterior_original"]
        ,levels =np.linspace(
            GRID["posterior_original"].min(),GRID["posterior_original"].max(), 30 )
    )
ax1.set_title("posterior on alpha, beta")
ax1.set_xlabel("alpha")
ax1.set_ylabel("beta")
plt.colorbar(plot1, ax =ax1)


plot2 = ax2.contourf(
        GRID["logit_mean"], GRID["log_sample_size"], GRID["posterior_transform"]
        #percentiles of the mode
        ,levels =GRID["posterior_transform"].max() * np.arange(.00,.95,.05)
    )
ax2.set_title("posterior on logit_mean, log_sample_size")
ax2.set_xlabel("logit_mean")
ax2.set_ylabel("log_sample_size")
plt.colorbar(plot2, ax =ax2)
plt.show()

# s = 2000 #number of simulated samples 
# posterior_alpha = marginal_of_alpha_on_grid(GRID["log_posterior_original"])
# draws_of_alpha = sample_from_empirical_dist(s, GRID["alpha"], posterior_alpha)

# draws_of_beta_given_alpha = np.zeros(s, dtype = float)

# for i, alpha_sample in enumerate(draws_of_alpha):
#     beta_dist = beta_on_grid_given_alpha(GRID, alpha_sample)
#     beta_sample = sample_from_empirical_dist(1, GRID["beta"], beta_dist )[0]
#     draws_of_beta_given_alpha[i] = beta_sample

# # #add jitter to sampled pairs
# # draws_of_alpha += uniform.rvs(size = s, scale = STEP_SIZE_IN_ALPHA)
# # draws_of_beta_given_alpha += uniform.rvs(size = s, scale = STEP_SIZE_IN_BETA)


# fig, (ax1, ax2) = plt.subplots(1,2)

# ax1.scatter(draws_of_alpha, draws_of_beta_given_alpha)
# ax1.set_xlabel("alpha")
# ax1.set_ylabel("beta")
# ax1.set_title(f"{s} simulations of hyperparameters")

# ax2.scatter(*transform_from_alpha_beta(draws_of_alpha, draws_of_beta_given_alpha))
# ax2.set_xlabel("logit_mean")
# ax2.set_ylabel("log_sample_size")
# ax2.set_title(f"{s} simulations of transformed hyperparameters")

# plt.show()

# # fig, (ax1, ax2) = plt.subplots(1,2, sharex = True)
# # ax1.bar(GRID["alpha"], posterior_alpha)
# # ax1.set_title("marginal unnormalised posterior of alpha")
# # values, heights = np.unique(draws_of_alpha, return_counts=True)
# # ax2.bar(values, heights)
# # ax2.set_title(f"frequency of {s} simulated draws from posterior")
# # plt.show()


# #sample theta
# rand_i = random.choice(range(s))
# print(rand_i)
# specific_alpha, specific_beta = draws_of_alpha[rand_i], draws_of_beta_given_alpha[rand_i]
# rand_i = random.choice(seq = range(71))
# specific_y, specific_n = dataset.loc[rand_i,]
# print(specific_y, specific_n)