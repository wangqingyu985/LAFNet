"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import pickle
import time as t
import serial as s
import numpy as np
from dataset.collect_dataset.modbus_and_CRC import modbus_cal
from utils.utils import plot_tactile_and_pressure_curve, plot_pressure_curve
import matplotlib
matplotlib.use('TKAgg')

"""
/dev/ttyUSB2: force sensor on pneumatic-driven soft gripper.
/dev/ttyUSB3: force sensor on e-glove.
"""
# pre-define:
negative_trigger, negative_recover = '01 05 01 01 FF 00 DC 06', '01 05 01 01 00 00 9D F6'
read_positive, read_byte, read_force_trigger = '01 03 03 02 00 01 25 8E', '', 'AA 01 00 00 00 00 00 FF'
fingertip_ch_digital = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
DESIRED_PRESSURE, MAX_PRESSURE, MAX_FORCE = 150, 150, 5.0
# information for building the dataset:
OBJ_NAME = 'tomato'  # 'kiwi' or 'tomato' or 'apple' or 'plum' or 'orange' or 'pear'
SAMPLE_NUMBER = '007'
FRAME_NUMBER = '010'
dir = '/media/wangqingyu/固态硬盘/ForceLearning/dataset/' + OBJ_NAME + '/'
filename = OBJ_NAME + '_sample_' + SAMPLE_NUMBER + '_frame_' + FRAME_NUMBER + '.pickle'
# serial define (air pump and tactile sensor):
air_pump = s.Serial(port='/dev/ttyUSB0', baudrate=9600)
force_com = s.Serial(port='/dev/ttyUSB2', baudrate=115200, timeout=0.1)


def set_desired_pressure(pressure):
    if pressure > MAX_PRESSURE:
        print('Dangerous air pressure! Please stop.')
        return
    else:
        air_pump.write(bytes.fromhex(modbus_cal(pressure_dec=pressure)))
        read_byte = air_pump.read(8).hex()


def positive_trigger():
    air_pump.write(bytes.fromhex('01 05 01 00 FF 00 8D C6'))
    read_byte = air_pump.read(8).hex()


def positive_recover():
    air_pump.write(bytes.fromhex('01 05 01 00 00 00 CC 36'))
    read_byte = air_pump.read(8).hex()


def read_air_pressure():
    air_pump.write(bytes.fromhex(read_positive))
    return int(air_pump.read(7).hex()[8:10], 16)


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
    return (fingertip_ch_digital * (5 / 3366) - (1000 / 561)).reshape(4, 4)


def save_data(tactile_data_final, air_pressure_final):
    data = {
        'obj_name': OBJ_NAME,
        'tactile_data': tactile_data_final,
        'air_pressure': air_pressure_final
    }
    with open(dir + filename, 'wb') as file:
        pickle.dump(data, file)


def main():
    set_desired_pressure(DESIRED_PRESSURE)
    t.sleep(0.1)
    positive_trigger()
    force_com.write(bytes.fromhex(read_force_trigger))  # initialize force sensor
    start_time = t.time()
    tactile_data, air_pressure = [], []
    while True:
        tactile_data.append(read_tactile_data() / MAX_FORCE)
        air_pressure.append(read_air_pressure() / MAX_PRESSURE)
        if t.time() - start_time > 4:
            break
    positive_recover()
    while True:
        tactile_data.append(read_tactile_data() / MAX_FORCE)
        air_pressure.append(read_air_pressure() / MAX_PRESSURE)
        if t.time() - start_time > 6:
            break
    if (len(tactile_data) > 150) and (len(air_pressure) > 150):
        tactile_data, air_pressure = tactile_data[:150], air_pressure[:150]
        tactile_data_final = np.array(tactile_data)
        air_pressure_final = np.array(air_pressure)
        save_data(tactile_data_final, air_pressure_final)
    else:
        raise Exception("Serial communication error! Please try to collect data again.")
    plot_tactile_and_pressure_curve(np.max(a=tactile_data_final, axis=(-2, -1)), air_pressure_final)
    # plot_pressure_curve(air_pressure_final)


if __name__ == "__main__":
    main()
