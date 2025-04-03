"""
created by Qingyu Wang
06.11.2024  14:16
email: 12013027@zju.edu.cn; qingyu.wang@uni-hamburg.de
"""
import numpy as np
import matplotlib.pyplot as plt

x = [2, 4, 6, 8, 10, 12, 14]
labels = ['Apple', 'Kiwi', 'Orange', 'Pear', 'Plum', 'Tomato', 'All']
apple = np.array([-0.0081, -0.0724, 0.0273, 0.0235, 0.0008, -0.0125, -0.0587, 0.0187, 0.0242, 0.0316]) * 2
kiwi = np.array([0.0543, -0.1075, 0.0376, 0.0889, -0.0775, 0.0536, -0.0326, 0.0911, -0.0419, 0.0997]) * 2
orange = np.array([-0.0357, -0.0284, -0.0650, -0.0794, 0.0167, 0.0614, 0.0297, -0.0084, 0.1223, 0.0112]) * 2
pear = np.array([0.0386, 0.0588, -0.0131, -0.0157, -0.0572, -0.0144, 0.0736, 0.0323, -0.0494, 0.0427]) * 2
plum = np.array([0.0735, 0.0235, -0.0055, 0.0561, 0.0192, 0.0306, 0.0485, 0.0028, -0.0151, -0.0248]) * 2
tomato = np.array([0.0330, -0.0436, 0.0079, 0.0502, 0.0790, 0.0882, -0.0118, 0.0190, 0.0410, 0.0473]) * 2
all_fruit = np.array([-0.0081, -0.0724, 0.0273, 0.0235, 0.0008, -0.0125, -0.0587, 0.0187, 0.0242, 0.0316, 0.0543, -0.1075, 0.0376, 0.0889, -0.0775, 0.0536, -0.0326, 0.0911, -0.0419, 0.0997, -0.0357, -0.0284, -0.0650, -0.0794, 0.0167, 0.0614, 0.0297, -0.0084, 0.1223, 0.0112, 0.0386, 0.0588, -0.0131, -0.0157, -0.0572, -0.0144, 0.0736, 0.0323, -0.0494, 0.0427, 0.0735, 0.0235, -0.0055, 0.0561, 0.0192, 0.0306, 0.0485, 0.0028, -0.0151, -0.0248, 0.0330, -0.0436, 0.0079, 0.0502, 0.0790, 0.0882, -0.0118, 0.0190, 0.0410, 0.0473]) * 2

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

plt.ylabel("Steady-state Error $e_{ss}$ (N)", fontdict=axis_font)
plt.xlim(1, 15)
plt.ylim(-0.3, 0.3)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.show()
