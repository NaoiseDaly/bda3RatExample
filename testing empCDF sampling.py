from at1 import *
import matplotlib.pyplot as plt

vals = np.array(range(1,5+1))
dist = np.array([.1,.3,.3,.1,.2])
ind_draws = [sample_from_empirical_dist(dist) for _ in range(20000)]
print(np.unique(vals[ind_draws], return_counts=True))


fig, (ax1, ax2) = plt.subplots(1,2)
ax1.bar(vals, dist)
ax1.set_title("theoretic")
ax2.bar(*np.unique(vals[ind_draws], return_counts=True))
ax2.set_title("simulated")
plt.show()