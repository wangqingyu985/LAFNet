"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 4, 7, 10]
labels = ['AP', 'Half TD+\nAP', 'TD+\nAP', 'TD']
ap = np.array([1.9933667182922363, 1.5413594245910645, 0.8601728081703186, 1.117319107055664, 0.2615359425544739, 0.8021959662437439, 1.1171513795852661, 1.0948255062103271, 0.7281315326690674, 1.117154598236084, 0.6015032529830933, 0.7992687821388245, 0.5157148838043213])
td_half_ap = np.array([0.8142796158790588, 0.857255756855011, 0.0054970383644104, 0.07242381572723389, 0.54433673620224, 0.5526155233383179, 2.027172803878784, 0.07026612758636475, 1.3394131660461426, 1.6235747337341309])
td_ap = np.array([0.3105580806732178, 0.17527878284454346, 1.667742133140564, 0.32495230436325073, 0.588422417640686, 0.16408830881118774, 0.45442670583724976, 1.0290501117706299])
td = np.array([0.6759342551231384, 1.7577112913131714, 0.21101564168930054, 0.6453919410705566, 1.9321162700653076, 0.9898775815963745, 0.7657542824745178, 0.06491035223007202, 2.0249977111816406, 0.6858238577842712, 0.9177044034004211, 0.596621036529541])

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

