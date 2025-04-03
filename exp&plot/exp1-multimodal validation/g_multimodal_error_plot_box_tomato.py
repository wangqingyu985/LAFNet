"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 4, 7, 10]
labels = ['AP', 'Half TD+\nAP', 'TD+\nAP', 'TD']
ap = np.array([0.24899408221244812, 0.15008178651332855, 0.6945697069168091, 0.2712186574935913, 0.26115546882152557, 0.3437499403953552, 0.27965996861457825, 0.2580730825662613, 0.18184509873390198, 0.29397027790546417, 0.16158588230609894, 0.3473496437072754, 0.4424191117286682, 0.2142711192369461, 0.260587078332901, 0.27894943952560425, 0.328541497886180878, 0.6903952956199646, 0.5406138896942139, 0.22557992935180664])
td_half_ap = np.array([0.2594826579093933, 0.31430983543395996, 0.25602079033851624, 0.3563232719898224, 0.14609582722187042, 0.19637385606765747, 0.18736675381660461, 0.031243711709976196, 0.15535816550254822, 0.7481955289840698])
td_ap = np.array([0.15974417328834534, 0.11481913328170776, 0.2128763496875763, 0.10679841041564941, 0.36849603056907654, 0.06625309586524963, 0.19176051020622253, 0.11769124269485474, 0.11862712502479553, 0.0968419760465622])
td = np.array([0.29134061634540558, 0.6269339323043823, 0.19461149573326111, 0.3292163014411926, 0.2299221158027649, 0.19521154463291168, 0.14349276423454285, 0.14813835024833679, 0.14933841526508331, 0.348881545662879944])

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
