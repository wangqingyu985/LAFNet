"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import torch
import torch.nn as nn


class LAFNet(nn.Module):
    def __init__(self, num_layers_transformer, num_layers_lstm, tactile_modal, full_tactile_modal, pressure_modal):
        super(LAFNet, self).__init__()

        self.num_layers_transformer = num_layers_transformer
        self.num_layers_lstm = num_layers_lstm
        self.tactile_modal = tactile_modal
        self.full_tactile_modal = full_tactile_modal
        self.pressure_modal = pressure_modal

        # tactile
        if self.tactile_modal:
            self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=1, stride=1, padding=0)
            self.conv3d2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.relu_tactile = nn.ReLU()
            if self.full_tactile_modal:
                self.pool_tactile = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
            else:
                self.pool_tactile = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.flatten_tactile = nn.Flatten(start_dim=-3, end_dim=-1)
            self.lstm_tactile = LSTMModel(input_size=150, hidden_size=150, output_size=150,
                                          num_layers=self.num_layers_lstm)

        # air pressure
        if self.pressure_modal:
            self.conv1d1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1, stride=1, padding=0)
            self.conv1d2 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.relu_pressure = nn.ReLU()
            self.lstm_pressure = LSTMModel(input_size=150, hidden_size=150, output_size=150,
                                           num_layers=self.num_layers_lstm)

        # transformer
        self.fc1_fusion = nn.Linear(in_features=(self.tactile_modal * 150 + self.pressure_modal * 150), out_features=32)
        self.relu_fusion = nn.ReLU()
        encoder_layers = nn.TransformerEncoderLayer(d_model=32, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers,
                                                         num_layers=self.num_layers_transformer)

        # output
        self.fc2_fusion = nn.Linear(32, 1)
        self.to(torch.float64)

    def forward(self, input_tactile, input_pressure):
        input_tactile = input_tactile.view(-1, 1, 150, 4, 4)  # [B, C=1, D=150, H=4, W=4]
        if not self.full_tactile_modal:
            input_tactile = input_tactile[:, :, :, 1:3, 1:3]  # [B, C=1, D=150, H=2, W=2]
        input_pressure = input_pressure.view(-1, 1, 150)  # [B, C=1, W=150]

        # tactile
        if self.tactile_modal:
            tactile_feature = self.conv3d2(self.relu_tactile(self.conv3d1(input_tactile))) + input_tactile
            # [B, C=1, D=150, H=4, W=4]
            tactile_feature = self.relu_tactile(tactile_feature)  # [B, C=1, D=150, H=4, W=4]
            tactile_feature = self.pool_tactile(tactile_feature)  # [B, C=1, D=150, H=1, W=1]
            tactile_feature = self.flatten_tactile(tactile_feature)  # [B, C=1, 150]
            tactile_feature = self.lstm_tactile(tactile_feature)  # [B, C=1, 150]

        # air pressure
        if self.pressure_modal:
            pressure_feature = self.conv1d2(self.relu_pressure(self.conv1d1(input_pressure))) + input_pressure
            # [B, C=1, W=150]
            pressure_feature = self.relu_pressure(pressure_feature)  # [B, C=1, W=150]
            pressure_feature = self.lstm_pressure(pressure_feature)  # [B, C=1, W=150]

        # fusion
        if self.tactile_modal and self.pressure_modal:
            fusion_feature = torch.cat(tensors=[tactile_feature, pressure_feature], dim=-1)  # [B, C=1, W=300]
        elif self.tactile_modal and not self.pressure_modal:
            fusion_feature = tactile_feature  # [B, C=1, W=150]
        else:
            fusion_feature = pressure_feature  # [B, C=1, W=150]

        # transformer
        fusion_feature = self.fc1_fusion(fusion_feature)  # [B, C=1, W=32]
        fusion_feature = self.relu_fusion(fusion_feature)  # [B, C=1, W=32]
        fusion_feature = self.transformer_encoder(fusion_feature)  # [B, C=1, W=32]

        # output adaptive grasping force
        output_feature = self.fc2_fusion(fusion_feature)  # [B, C=1, W=1]
        output = output_feature.view(-1, 1)  # [B, output=1]

        return output


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch, seq_len, hidden_size)
        return out


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = LAFNet(num_layers_transformer=2, num_layers_lstm=4,
                   tactile_modal=True, full_tactile_modal=True, pressure_modal=True)
    input_tactile_tensor = torch.randn(1, 1, 150, 4, 4)  # [B, C, 150, 4, 4]
    input_pressure_tensor = torch.randn(1, 1, 150)  # [B, C, 150]
    params = count_parameters(m=model)
    print(f"Model Parameters: {params}")
