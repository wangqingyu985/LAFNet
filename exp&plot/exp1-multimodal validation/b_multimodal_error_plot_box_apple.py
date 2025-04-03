"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 4, 7, 10]
labels = ['AP', 'Half TD+\nAP', 'TD+\nAP', 'TD']
ap = np.array([0.8548638224601746, 0.5503806471824646, 0.9191375970840454, 0.03366321325302124, 0.869838297367096, 0.7933741807937622, 0.6360921263694763, 0.5619433522224426, 0.5612123012542725, 1.151289701461792, 1.3888945579528809, 0.6334954500198364, 0.3626987338066101, 0.5530506372451782, 0.42599260807037354, 1.5051188468933105, 0.10865867137908936, 0.9207028150558472, 0.5186918377876282, 1.719978094100952])
td_half_ap = np.array([1.6777691841125488, 0.24904131889343262, 0.5230468511581421, 0.3860282897949219, 0.11213570833206177, 2.093932867050171, 0.38200557231903076, 0.3887590765953064, 0.9174260497093201])
td_ap = np.array([0.27907490730285645, 1.369299292564392, 0.01729518175125122, 0.48559337854385376, 0.3172290325164795, 1.1713606119155884, 0.2271360158920288, 0.5379590392112732, 0.06738841533660889, 0.721057653427124])
td = np.array([0.24656802415847778, 0.8018738031387329, 1.0832220315933228, 0.9818771481513977, 0.6601974368095398, 1.1527929306030273, 0.305497944355011, 0.5253100395202637, 0.988839864730835, 1.5568509101867676])

all_data = [ap, td_half_ap, td_ap, td]

plt.figure(figsize=(6, 4.5), dpi=600)

axis_font = {'weight': 'bold', 'size': 14}
title_font = {'weight': 'bold', 'size': 15}
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.weight"] = "bold"

colors_pale = ['#66c5cc', '#f6cf71', '#f89c74', '#dcb0f2']

box = plt.boxplot(all_data, patch_artist=True, positions=[1, 4, 7, 10],
                  showmeans=False, widths=1.4, sym="k+")
for patch, color in zip(box['boxes'], colors_pale):
    patch.set_facecolor(color)

plt.ylabel("Error (N)", fontdict=axis_font)
plt.xlim(0, 11)
plt.ylim(0, 5)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.show()

