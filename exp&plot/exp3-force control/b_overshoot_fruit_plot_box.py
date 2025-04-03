"""
created by Qingyu Wang
06.11.2024  14:16
email: 12013027@zju.edu.cn; qingyu.wang@uni-hamburg.de
"""
import numpy as np
import matplotlib.pyplot as plt

x = [2, 4, 6, 8, 10, 12, 14]
labels = ['Apple', 'Kiwi', 'Orange', 'Pear', 'Plum', 'Tomato', 'All']
apple = np.array([126.82, 130.50, 128.87, 150.34, 138.60, 131.86, 137.09, 123.51, 122.32, 129.49]) * 0.5
kiwi = np.array([43.37, 23.81, 53.09, 36.80, 19.27, 21.02, 23.74, 26.85, 29.91, 27.00]) * 0.5
orange = np.array([80.42, 83.41, 80.22, 95.20, 103.42, 96.31, 68.49, 79.85, 67.02, 62.18]) * 0.5
pear = np.array([49.71, 48.29, 77.11, 78.43, 105.20, 105.41, 98.27, 106.65, 91.62, 78.39]) * 0.5
plum = np.array([73.51, 71.07, 63.36, 62.61, 68.65, 72.74, 64.57, 62.91, 79.53, 75.03]) * 0.5
tomato = np.array([69.08, 58.36, 54.68, 55.02, 59.68, 58.82, 67.26, 45.30, 55.85, 51.65]) * 0.5
all_fruit = np.array([126.82, 130.50, 128.87, 150.34, 138.60, 131.86, 137.09, 123.51, 122.32, 129.49, 43.37, 23.81, 53.09, 36.80, 19.27, 21.02, 23.74, 26.85, 29.91, 27.00, 80.42, 83.41, 80.22, 95.20, 103.42, 96.31, 68.49, 79.85, 67.02, 62.18, 49.71, 48.29, 77.11, 78.43, 105.20, 105.41, 98.27, 106.65, 91.62, 78.39, 73.51, 71.07, 63.36, 62.61, 68.65, 72.74, 64.57, 62.91, 79.53, 75.03, 69.08, 58.36, 54.68, 55.02, 59.68, 58.82, 67.26, 45.30, 55.85, 51.65]) * 0.5

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

plt.ylabel("Maximum Overshoot $\delta$ (%)", fontdict=axis_font)
plt.xlim(1, 15)
plt.ylim(0, 100)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.show()
