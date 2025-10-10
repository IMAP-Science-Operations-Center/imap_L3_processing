from matplotlib import pyplot as plt
from spacepy.pycdf import CDF

de = CDF("data/imap/hit/l3/2010/01/imap_hit_l3_direct-events_20100105_v001.cdf")

print(de)

xs = []
ys = []

for i in range(0, len(de['energy'][...])):

    if de['range'][i] == 3 and de['side'][i] == 1 and de['stim_tag'][i] != 1:
        ys.append(de['delta_e'][i])
        xs.append(de['e_prime'][i])

#
# for delta_e, e_prime, d_range in zip(de["delta_e"], de["e_prime"], de["range"], s):
#     if d_range == 3 and :
#         xs.append(e_prime)
#         ys.append(delta_e)

plt.scatter(xs, ys)
plt.xscale('log')
plt.yscale('log')
plt.show()
