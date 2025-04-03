"""
created by Qingyu Wang
06.11.2024  14:16
email: 12013027@zju.edu.cn; qingyu.wang@uni-hamburg.de
"""
import numpy as np
import matplotlib.pyplot as plt

x = [2, 4, 6, 8, 10, 12, 14]
labels = ['Apple', 'Kiwi', 'Orange', 'Pear', 'Plum', 'Tomato', 'All']
apple = np.array([2.8177, 2.8038, 3.4918, 4.3481, 2.8209, 2.8146, 4.6069, 4.6155, 2.8192, 4.6740])
kiwi = np.array([1.6361, 1.3380, 1.3449, 1.3468, 1.3444, 1.8380, 1.8094, 1.7831, 1.3459, 1.8642])
orange = np.array([1.1687, 1.1608, 1.3913, 1.4778, 1.3536, 2.5660, 3.5859, 3.4703, 3.6388, 3.9076])
pear = np.array([3.8797, 3.9318, 4.2277, 4.3153, 4.3385, 4.3390, 4.1709, 4.3175, 4.0685, 4.1343])
plum = np.array([1.3317, 1.3332, 1.3794, 1.1626, 2.8211, 1.3553, 2.5269, 2.8149, 2.9341, 2.5449])
tomato = np.array([2.5335, 3.2534, 2.8192, 3.5152, 3.5147, 3.5795, 1.5885, 3.5379, 3.5793, 3.4641])
all_fruit = np.array([2.8177, 2.8038, 3.4918, 4.3481, 2.8209, 2.8146, 4.6069, 4.6155, 2.8192, 4.6740, 1.6361, 1.3380, 1.3449, 1.3468, 1.3444, 1.8380, 1.8094, 1.7831, 1.3459, 1.8642, 1.1687, 1.1608, 1.3913, 1.4778, 1.3536, 2.5660, 3.5859, 3.4703, 3.6388, 3.9076, 3.8797, 3.9318, 4.2277, 4.3153, 4.3385, 4.3390, 4.1709, 4.3175, 4.0685, 4.1343, 1.3317, 1.3332, 1.3794, 1.1626, 2.8211, 1.3553, 2.5269, 2.8149, 2.9341, 2.5449, 2.5335, 3.2534, 2.8192, 3.5152, 3.5147, 3.5795, 1.5885, 3.5379, 3.5793, 3.4641])

all_data = [apple, kiwi, orange, pear, plum, tomato, all_fruit]

plt.figure(figsize=(7, 4.5), dpi=600)

axis_font = {'weight': 'bold', 'size': 14}
title_font = {'weight': 'bold', 'size': 15}
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.weight"] = "bold"

colors_pale = ['#fbb4ae', '#7fc97f', '#fdc086', '#ffe281', '#beaed4', '#fb9a99', '#7cadee']

box = plt.boxplot(all_data, patch_artist=True, positions=[2, 4, 6, 8, 10, 12, 14],
                  showmeans=False, widths=1.2, sym="k+")
for patch, color in zip(box['boxes'], colors_pale):
    patch.set_facecolor(color)

plt.ylabel("Predicted Force (N)", fontdict=axis_font)
plt.xlim(1, 15)
plt.ylim(0, 5)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.show()

