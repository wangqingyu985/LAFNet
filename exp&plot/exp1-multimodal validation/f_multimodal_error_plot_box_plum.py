"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1, 4, 7, 10]
labels = ['AP', 'Half TD+\nAP', 'TD+\nAP', 'TD']
ap = np.array([0.3210999071598053, 0.31169867515563965, 0.401313841342926, 0.02995312213897705, 0.15918664634227753, 0.2691093683242798, 0.25228098034858704, 0.07402785122394562, 0.18146827816963196, 0.46370241045951843])
td_half_ap = np.array([0.22496774792671204, 0.1861044019460678, 0.3349494934082031, 0.08339546620845795, 0.004543587565422058, 0.43648993968963623, 0.19172802567481995, 0.22377841174602509, 0.25567835569381714, 0.1241864264011383])
td_ap = np.array([0.024425163865089417, 0.027610063552856445, 0.13373591005802155, 0.22765956819057465, 0.32759130001068115, 0.20272515714168549, 0.1049429178237915, 0.34997648000717163, 0.1379101723432541, 0.01920424401760101])
td = np.array([0.027146637439727783, 0.27131956815719604, 0.1851944625377655, 0.3181433081626892, 0.13851478695869446, 0.15074864029884338, 0.11303223669528961, 0.19049569964408875, 0.41865304112434387, 0.09249672293663025])

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

