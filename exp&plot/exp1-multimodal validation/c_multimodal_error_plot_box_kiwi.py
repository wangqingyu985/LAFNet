"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 4, 7, 10]
labels = ['AP', 'Half TD+\nAP', 'TD+\nAP', 'TD']
ap = np.array([0.49816638231277466, 0.61700040102005, 0.24419806897640228, 1.3259166479110718, 0.6276065707206726, 0.6853359937667847, 0.7494269609451294, 0.17305530607700348, 1.5515432357788086, 0.23100651800632477, 0.2575659155845642])
td_half_ap = np.array([0.5531436204910278, 0.7591741681098938, 0.16830861568450928, 1.1867611408233643, 0.6980863213539124, 0.02713508903980255, 0.20638488233089447, 0.27668261528015137, 0.17340026795864105, 0.2945033311843872, 0.10983087122440338, 0.35096079111099243, 1.2137244939804077, 0.5410674810409546])
td_ap = np.array([0.05531921982765198, 0.10621003806591034, 0.029360204935073853, 0.059234052896499634, 0.06768882274627686, 0.3154356777667999, 0.31535422801971436, 0.09288027882575989, 0.5885671377182007, 0.43172264099121094, 0.2603474259376526, 0.6557540893554688, 0.15564262866973877, 0.5519087314605713, 0.37314778566360474, 0.6543805599212646])
td = np.array([1.2290630340576172, 0.8121346235275269, 0.5479931235313416, 0.5859761238098145, 0.2697342038154602, 0.04246369004249573, 0.5344557166099548, 0.5137481689453125, 0.12020021677017212, 0.11105149984359741, 0.5519684553146362, 0.6016288995742798, 0.35220563411712646, 0.277010440826416])

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


