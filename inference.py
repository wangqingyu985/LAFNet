"""
Created by Mr. Qingyu Wang at 14:33 22.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import torch
import argparse
import time as t
import serial as s
import numpy as np
from model.LAFNet import LAFNet
from utils.utils import show_force_and_time
from dataset.collect_dataset.input_data_record import read_tactile_data, read_air_pressure,\
    positive_trigger, positive_recover, set_desired_pressure


DESIRED_PRESSURE, MAX_PRESSURE, MAX_FORCE, read_force_trigger = 150, 150, 5.0, 'AA 01 00 00 00 00 00 FF'
force_com = s.Serial(port='/dev/ttyUSB2', baudrate=115200, timeout=0.1)

parser = argparse.ArgumentParser(description='Demo for self-adaptive grasping force learning network: LAFNet.')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/model_best.pth',
                    help='Checkpoint direction for loading.')
args = parser.parse_args()


if torch.cuda.is_available():
    print("CUDA is available")
    print("Device name:", torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)


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
    tactile_data, air_pressure = tactile_data[:150], air_pressure[:150]
    tactile_data_final = np.array(tactile_data)
    air_pressure_final = np.array(air_pressure)

    model = LAFNet(num_layers_transformer=2, num_layers_lstm=4, tactile_modal=True, full_tactile_modal=True,
                   pressure_modal=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint_dir))
    model.eval()
    with torch.no_grad():
        tactile_data = torch.from_numpy(tactile_data_final).to(device)
        air_pressure = torch.from_numpy(air_pressure_final).to(device)
        start_time = t.time()
        adaptive_force_pred = float(model(tactile_data, air_pressure).detach().cpu().float()) * 5
        end_time = t.time()
        test_time = abs(end_time - start_time) * 1000
    show_force_and_time(adaptive_force_pred=adaptive_force_pred, test_time=test_time)


if __name__ == '__main__':
    main()
