"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import os
import cv2
import math
import torch
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss_curve(train_loss_history, val_loss_history):
    plt.figure(figsize=(7, 5), dpi=600)
    axis_font = {'weight': 'bold', 'size': 14}
    title_font = {'weight': 'bold', 'size': 15}
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["font.weight"] = "bold"
    plt.plot(train_loss_history, label='Training Loss', color='#b01f24')
    plt.plot(val_loss_history, label='Validation Loss', color='#003f88')
    plt.title(label='Training and Validation Loss Over Epochs', fontdict=title_font)
    plt.xlabel(xlabel='Epochs', fontdict=axis_font)
    plt.ylabel(ylabel='Loss', fontdict=axis_font)
    plt.legend(loc='upper right')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    plt.show()


def plot_error_curve(val_error_list):
    plt.figure(figsize=(7, 5), dpi=600)
    axis_font = {'weight': 'bold', 'size': 14}
    title_font = {'weight': 'bold', 'size': 15}
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["font.weight"] = "bold"
    plt.plot(val_error_list, label='Validation Error', color='#b01f24')
    plt.title(label='Validation Error Over Epochs', fontdict=title_font)
    plt.xlabel(xlabel='Epochs', fontdict=axis_font)
    plt.ylabel(ylabel='Error (N)', fontdict=axis_font)
    plt.legend(loc='upper right')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    plt.show()


def plot_tactile_and_pressure_curve(tactile_data_final, air_pressure_final):
    plt.figure(figsize=(12, 12), dpi=600)
    axis_font = {'weight': 'bold', 'size': 14}
    title_font = {'weight': 'bold', 'size': 15}
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["font.weight"] = "bold"
    plt.plot(tactile_data_final, label='Tactile', color='#b01f24')
    plt.plot(air_pressure_final, label='Air Pressure', color='#003f88')
    plt.xlabel(xlabel='Sequence', fontdict=axis_font)
    plt.ylabel(ylabel='Normalized Value', fontdict=axis_font)
    plt.legend(loc='upper right')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    plt.show()


def plot_pressure_curve(air_pressure_final):
    fig = plt.figure(figsize=(12, 12), dpi=600)
    axis_font = {'weight': 'bold', 'size': 14}
    title_font = {'weight': 'bold', 'size': 15}
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["font.weight"] = "bold"
    plt.plot(air_pressure_final, label='Air Pressure', color='#003f88')
    plt.xlabel(xlabel='Sequence', fontdict=axis_font)
    plt.ylabel(ylabel='Normalized Value', fontdict=axis_font)
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    plt.show()


def read_pickle(dir='/media/wangqingyu/固态硬盘/ForceLearning/dataset/apple/apple_sample_002_frame_007.pickle'):
    with open(file=dir, mode='rb') as file:
        data = pickle.load(file)
    return data


def test_gpu():
    print("PyTorch Version: ", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available")


def plot_ground_truth_density_map(data_dir):
    apple, kiwi, orange, pear, plum, tomato = [], [], [], [], [], []
    for file in os.listdir(data_dir + 'apple/'):
        with open(file=(data_dir + 'apple/' + file), mode='rb') as f:
            apple.append(pickle.load(f)['ground_truth'] * 5)
    for file in os.listdir(data_dir + 'kiwi/'):
        with open(file=(data_dir + 'kiwi/' + file), mode='rb') as f:
            kiwi.append(pickle.load(f)['ground_truth'] * 5)
    for file in os.listdir(data_dir + 'orange/'):
        with open(file=(data_dir + 'orange/' + file), mode='rb') as f:
            orange.append(pickle.load(f)['ground_truth'] * 5)
    for file in os.listdir(data_dir + 'pear/'):
        with open(file=(data_dir + 'pear/' + file), mode='rb') as f:
            pear.append(pickle.load(f)['ground_truth'] * 5)
    for file in os.listdir(data_dir + 'plum/'):
        with open(file=(data_dir + 'plum/' + file), mode='rb') as f:
            plum.append(pickle.load(f)['ground_truth'] * 5)
    for file in os.listdir(data_dir + 'tomato/'):
        with open(file=(data_dir + 'tomato/' + file), mode='rb') as f:
            tomato.append(pickle.load(f)['ground_truth'] * 5)
    all = apple + kiwi + orange + pear + plum + tomato
    plt.figure(figsize=(7, 5), dpi=600)
    axis_font = {'weight': 'bold', 'size': 14}
    title_font = {'weight': 'bold', 'size': 15}
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["font.weight"] = "bold"
    sns.kdeplot(apple, shade=True, color="#fbb4ae", label="Apple", linewidth=1.6, alpha=0.12)
    sns.kdeplot(kiwi, shade=True, color="#7fc97f", label="Kiwi", linewidth=1.6, alpha=0.12)
    sns.kdeplot(orange, shade=True, color="#fdc086", label='Orange', linewidth=1.6, alpha=0.12)
    sns.kdeplot(pear, shade=True, color="#ffe281", label='Pear', linewidth=1.6, alpha=0.12)
    sns.kdeplot(plum, shade=True, color="#beaed4", label='Plum', linewidth=1.6, alpha=0.12)
    sns.kdeplot(tomato, shade=True, color="#fb9a99", label='Tomato', linewidth=1.6, alpha=0.12)
    sns.kdeplot(all, shade=True, color="#7cadee", label='All', linewidth=2, alpha=0.25)
    plt.xlabel(xlabel='Grasping Force (Ground Truth)', fontdict=axis_font)
    plt.ylabel(ylabel='Density', fontdict=axis_font)
    plt.legend(loc='upper right', fontsize=15)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    plt.show()
    print(np.mean(tomato), np.std(tomato))


def show_force_and_time(adaptive_force_pred, test_time):
    Img = np.zeros((1000, 1200, 3), dtype=np.uint8)
    Img[:] = (255, 255, 255)
    cv2.putText(img=Img, text='Adaptive Grasping Force:',
                org=(25, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.8, color=(136, 63, 0), thickness=8)
    cv2.putText(img=Img, text=' %.4f N' % adaptive_force_pred,
                org=(25, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.8, color=(136, 63, 0), thickness=8)
    cv2.putText(img=Img, text='Elapsed time:',
                org=(25, 350), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.8, color=(136, 63, 0), thickness=8)
    cv2.putText(img=Img, text=' %.4f ms' % test_time,
                org=(25, 450), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.8, color=(136, 63, 0), thickness=8)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname='result', height=1000, width=1200)
    cv2.imshow(winname='result', mat=Img)
    cv2.waitKey(delay=5000)


def move_to_pose(panda, pose, speed_factor):
    move_pose = panda.get_pose()
    move_pose[0, 3], move_pose[1, 3], move_pose[2, 3] = pose[0], pose[1], pose[2]
    R = eulerAnglesToRotationMatrix(np.array([pose[3], pose[4], pose[5]]))
    for i in range(0, 3):
        for j in range(0, 3):
            move_pose[i, j] = R[i, j]
    panda.move_to_pose(position=move_pose, speed_factor=speed_factor)


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


if __name__ == '__main__':
    data_dir = '/media/wangqingyu/固态硬盘/ForceLearning/dataset/'
    plot_ground_truth_density_map(data_dir)
