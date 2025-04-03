"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import pickle
import time as t
import serial as s
import numpy as np


"""
/dev/ttyUSB2: force sensor on pneumatic-driven soft gripper.
/dev/ttyUSB3: force sensor on e-glove.
"""
# pre-define:
fingertip_ch_digital = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
MAX_FORCE, MAX_FIRMNESS = 5.0, 100.0
read_force_trigger = 'AA 01 00 00 00 00 00 FF'
# information for building the dataset:
OBJ_NAME = 'tomato'  # 'kiwi' or 'tomato' or 'apple' or 'plum' or 'orange' or 'pear'
SAMPLE_NUMBER = '007'
FRAME_NUMBER = '010'
FIRMNESS = (60 / MAX_FIRMNESS)
dir = '/media/wangqingyu/固态硬盘/ForceLearning/dataset/' + OBJ_NAME + '/'
filename = OBJ_NAME + '_sample_' + SAMPLE_NUMBER + '_frame_' + FRAME_NUMBER + '.pickle'
# serial define (tactile sensor):
force_com = s.Serial(port='/dev/ttyUSB3', baudrate=115200, timeout=0.1)


def read_tactile_data():
    ReadLine = force_com.read(38).hex()
    while True:
        if len(ReadLine) != 76 or (ReadLine[0] + ReadLine[1] + ReadLine[2] + ReadLine[3]) != '2400':
            force_com.close()
            force_com.open()
            ReadLine = force_com.read(38).hex()
        else:
            break
    for i in range(0, 8):
        fingertip_ch_digital[i] = int('0x' + ReadLine[4 * i + 4] + ReadLine[4 * i + 5], 16) * 256 + int(
            '0x' + ReadLine[4 * i + 6] + ReadLine[4 * i + 7], 16)
    for i in range(8, 16):
        fingertip_ch_digital[i] = int('0x' + ReadLine[4 * i + 10] + ReadLine[4 * i + 11], 16) * 256 + int(
            '0x' + ReadLine[4 * i + 12] + ReadLine[4 * i + 13], 16)
    return (fingertip_ch_digital * (5 / 3366) - (1000 / 561)).max()


def save_data(tactile_data_ground_truth):
    with open(file=dir + filename, mode='rb') as file:
        data = pickle.load(file)
    data['ground_truth'] = tactile_data_ground_truth
    data['firmness'] = FIRMNESS
    print('Normalized ground truth: ', tactile_data_ground_truth)
    print('Normalized firmness: ', FIRMNESS)
    with open(dir + filename, 'wb') as file:
        pickle.dump(data, file)


def main():
    force_com.write(bytes.fromhex(read_force_trigger))  # initialize force sensor
    start_time = t.time()
    tactile_data = []
    while True:
        tactile_data.append(read_tactile_data() / MAX_FORCE)
        if t.time() - start_time > 5:
            break
    tactile_data_ground_truth = max(tactile_data)
    save_data(tactile_data_ground_truth)


if __name__ == "__main__":
    main()
