"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 4, 7, 10]
labels = ['AP', 'Half TD+\nAP', 'TD+\nAP', 'TD']
ap = np.array([0.4396229684352875, 0.06997883319854736, 0.0505085289478302, 0.016212239861488342, 0.1409592479467392, 0.010435357689857483, 0.04026450216770172, 0.3343088924884796])
td_half_ap = np.array([0.0585617870092392, 0.5000775456428528, 0.11336587369441986, 0.15358969569206238, 0.06456874310970306, 0.23689821362495422, 0.04656262695789337, 0.2782905101776123])
td_ap = np.array([0.09134352207183838, 0.05001083016395569, 0.09575925767421722, 0.054473876953125, 0.04336044192314148, 0.258197546005249, 0.00599980354309082, 0.43956249952316284, 0.06801478564739227, 0.05091644823551178])
td = np.array([0.12210063636302948, 0.012583434581756592, 0.3907606601715088, 0.19701480865478516, 0.15731468796730042, 0.04735685884952545, 0.06768845021724701])

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

