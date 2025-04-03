"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import torch
import argparse
import time as t
from torch import nn
from model.LAFNet import LAFNet
from torch.utils.data import random_split, DataLoader
from dataloader.data_loader import PickleDatasetLoader
from utils.utils import plot_loss_curve, plot_error_curve


parser = argparse.ArgumentParser(description='Train a self-adaptive grasping force learning network: LAFNet.')
parser.add_argument('--tactile_modal', type=bool, default=True, help='Whether to take tactile modal as input.')
parser.add_argument('--full_tactile_modal', type=bool, default=True, help='Whether to take full tactile modal as input.')
parser.add_argument('--pressure_modal', type=bool, default=True, help='Whether to take air pressure modal as input.')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
parser.add_argument('--data_dir', type=str, default='./dataset/all/', help='Dataset direction for loading.')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Train ratio.')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio, rest for test.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--save_interval', type=int, default=1, help='Save interval of the weights as checkpoints.')
parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='Directory to save checkpoints.')
args = parser.parse_args()


if torch.cuda.is_available():
    print("CUDA is available")
    print("Device name:", torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)


def main():

    dataset = PickleDatasetLoader(data_dir=args.data_dir)
    train_size = int(args.train_ratio * len(dataset))
    val_size = int(args.val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_loader, validation_loader, test_loader = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(dataset=train_loader, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_loader, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=test_loader, batch_size=1, shuffle=False)

    model = LAFNet(num_layers_transformer=2, num_layers_lstm=4,
                   tactile_modal=args.tactile_modal, full_tactile_modal=args.full_tactile_modal,
                   pressure_modal=args.pressure_modal).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    train_loss_history, val_loss_history = [], []
    val_error_list, test_error_list = [], []
    val_error_best = 5.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for tactile_data, air_pressure, adaptive_force_gt in train_loader:
            optimizer.zero_grad()
            tactile_data = tactile_data.to(device)
            air_pressure = air_pressure.to(device)
            adaptive_force_gt = adaptive_force_gt.to(device)
            adaptive_force_pred = model(tactile_data, air_pressure)
            loss = loss_fn(adaptive_force_pred, adaptive_force_gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if (epoch + 1) % args.save_interval == 0:
            save_path = args.save_dir + f'model_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} to {save_path}")

        model.eval()
        val_loss, val_error = 0.0, 0.0
        with torch.no_grad():
            for tactile_data, air_pressure, adaptive_force_gt in validation_loader:
                tactile_data = tactile_data.to(device)
                air_pressure = air_pressure.to(device)
                adaptive_force_gt = adaptive_force_gt.to(device)
                adaptive_force_pred = model(tactile_data, air_pressure)
                val_loss += loss_fn(adaptive_force_pred, adaptive_force_gt).item()
                val_error += abs(adaptive_force_pred.detach().cpu().float() - adaptive_force_gt.detach().cpu().float()) * 5.0

        avg_train_loss, avg_val_loss = train_loss / len(train_loader), val_loss / len(validation_loader)
        val_error_list.append(val_error / len(validation_loader))
        if val_error_list[-1] < val_error_best:
            save_path = args.save_dir + 'model_best.pth'
            torch.save(model.state_dict(), save_path)
            best_epoch, val_error_best = epoch + 1, val_error_list[-1]
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Validation Loss={avg_val_loss:.4f}")

    print(f"Best model saved at epoch {best_epoch + 1}")
    model.load_state_dict(torch.load(args.save_dir + 'model_best.pth'))
    model.eval()
    test_error, test_time = [], []
    with torch.no_grad():
        for tactile_data, air_pressure, adaptive_force_gt in test_loader:
            tactile_data = tactile_data.to(device)
            air_pressure = air_pressure.to(device)
            adaptive_force_gt = adaptive_force_gt.to(device)
            start_time = t.time()
            adaptive_force_pred = model(tactile_data, air_pressure)
            end_time = t.time()
            test_error.append(float(abs(adaptive_force_pred.detach().cpu().float() - adaptive_force_gt.detach().cpu().float()) * 5.0))
            test_time.append(abs(end_time - start_time) * 1000)
    print('Error on test set:', test_error, ' N')
    print('Time on test set:', test_time, ' ms')
    plot_loss_curve(train_loss_history, val_loss_history)
    plot_error_curve(val_error_list)


if __name__ == '__main__':
    main()
