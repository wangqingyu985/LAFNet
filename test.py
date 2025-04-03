"""
Created by Mr. Qingyu Wang at 14:33 22.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import torch
import argparse
import time as t
import numpy as np
from model.LAFNet import LAFNet
from utils.utils import read_pickle

parser = argparse.ArgumentParser(description='Test a self-adaptive grasping force learning network: LAFNet.')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/model_best.pth',
                    help='Checkpoint direction for loading.')
parser.add_argument('--data_dir', type=str, default='./dataset/tomato/tomato_sample_004_frame_006.pickle',
                    help='Data direction for loading.')
args = parser.parse_args()


if torch.cuda.is_available():
    print("CUDA is available")
    print("Device name:", torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)


def main():
    model = LAFNet(num_layers_transformer=2, num_layers_lstm=4, tactile_modal=True, full_tactile_modal=True,
                   pressure_modal=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint_dir))
    model.eval()
    with torch.no_grad():
        data = read_pickle(dir=args.data_dir)
        tactile_data = torch.from_numpy(data['tactile_data']).to(device)
        air_pressure = torch.from_numpy(data['air_pressure']).to(device)
        adaptive_force_gt = torch.from_numpy(np.array([data['ground_truth']])).to(device)
        start_time = t.time()
        adaptive_force_pred = model(tactile_data, air_pressure)
        end_time = t.time()
        test_error = float(abs(adaptive_force_pred.detach().cpu().float() - adaptive_force_gt.detach().cpu().float()) * 5.0)
        test_time = abs(end_time - start_time) * 1000
    print('Error:', test_error, ' N')
    print('Elapsed time:', test_time, ' ms')


if __name__ == '__main__':
    main()
