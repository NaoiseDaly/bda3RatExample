from at1 import *
import pandas as pd
import matplotlib.pyplot as plt

vals = np.array(range(1,5+1))
dist = np.array([.1,.3,.3,.1,.2])
ind_draws = [sample_from_empirical_dist(dist) for _ in range(20)]
print(vals[ind_draws][:20])


fig, (ax1, ax2) = plt.subplots(1,2)
ax1.bar(vals, dist)
ax1.set_title("theoretic")
ax2.hist(ind_draws)
ax2.set_title("simulated")
plt.show()