from at1 import *
import matplotlib.pyplot as plt

vals = np.array(range(5))
dist = np.array([.1,.3,.3,.1,.2])


draws = sample_from_empirical_dist(2000, vals, dist)
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.bar(vals, dist)
ax1.set_title("theoretic")
ax2.bar(*np.unique(draws, return_counts=True))
ax2.set_title("simulated")
plt.show()