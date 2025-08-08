import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models


class StandardizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(StandardizeLayer, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class VGPNet(nn.Module):
    def __init__(
        self,
        imean=torch.tensor([0, 0, 0], dtype=torch.float64),
        istd=torch.tensor([1, 1, 1], dtype=torch.float64),
        sequence_length=5,  # 输入序列长度
        prediction_length=8,  # 输出预测长度
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        
        # LSTM处理卫星序列
        self.gnss_lstm = nn.LSTM(
            input_size=3,
            hidden_size=128,
            num_layers=2,
            batch_first=False,
            dropout=0.1,
            bidirectional=False
        )
            

        # 图像特征提取
        self.img_encoder = models.resnet18(pretrained=True)
        num_features_img = self.img_encoder.fc.in_features
        self.img_encoder.fc = nn.Linear(num_features_img, 64)
        
        # 序列图像特征提取
        self.img_lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=False,
            dropout=0.1,
            bidirectional=False
        )

        feature_dim = 256
        
        self.weights_bias_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        
        self.pred_weight_bias_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.prediction_length * 2),  # 预测长度的权重和偏置
        )
        
        self.ccffm = CCFFM(channels=64, heads=4)

    def forward(self, x_sequence, img_sequence):
        """
        前向传播
        Args:
            x_sequence: 输入序列 [sequence_length, sats_num, 3]
            img_sequence: 图像序列 [sequence_length, 3, 224, 224]
        Returns:
            weights: [sats_num, 1]  表示第sequence_length个时间步的权重
            biases: [sats_num, 1]  表示第sequence_length个时间步的偏置
            predictions: [prediction_length, lat, lon, alt]  表示预测的未来8个时间步的位置
        """
        # 获取sequence_length的最大卫星数
        max_sats_num = 0
        # last_sats_num = x_sequence[-1].shape[0]  # 最后一个时间步的卫星数
        for x in x_sequence:
            max_sats_num = max(max_sats_num, x.shape[0])  # 获取最大卫星数
        
        # pad x_sequence
        padded_x_sequence = []
        for x in x_sequence:
            if x.shape[0] < max_sats_num:
                padding = torch.zeros(max_sats_num - x.shape[0], 3, dtype=x.dtype, device=x.device)
                padded_x_sequence.append(torch.cat([x, padding], dim=0))
            else:
                padded_x_sequence.append(x)
        
        x_sequence = torch.stack(padded_x_sequence, dim=0)  # [sequence_length, max_sats_num, 3]

        # LSTM处理卫星序列
        lstm_out, _ = self.gnss_lstm(x_sequence)  # [sequence_length, max_sats_num, 128]

        
        # 序列图像特征提取
        for i in range(len(img_sequence)):
            img_sequence[i] = self.img_encoder(img_sequence[i]).squeeze(0)  # [64]
        img_sequence = torch.stack(img_sequence, dim=0)  # [sequence_length, 64]

        # LSTM处理图像序列
        img_lstm_out, _ = self.img_lstm(img_sequence) # [sequence_length, 128]
        
        # 取最后一个时间步的卫星特征和图像特征
        last_sat_features = lstm_out[-1]
        # if last_sat_features.shape[0] > last_sats_num:
            # last_sat_features = last_sat_features[:last_sats_num]
        
        last_img_features = img_lstm_out[-1]
        last_img_features = last_img_features.repeat(max_sats_num, 1)  # [last_sats_num, 128]
        
        # print(f"last_sat_features shape: {last_sat_features.shape}, last_img_features shape: {last_img_features.shape}")
        
        # 卫星特征和图像特征拼接
        combined_features = torch.cat([last_sat_features, last_img_features], dim=-1)
        # 计算index为4的权重和偏置
        weights_bias = self.weights_bias_head(combined_features)
        
        weights = torch.sigmoid(weights_bias[:, 0])
        weights = torch.clamp(weights, min=0, max=1)
        biases = F.leaky_relu(weights_bias[:, 1])
        
        
        # 计算未来8个时间步的权重和偏置
        pred_weights_bias = self.pred_weight_bias_head(combined_features)
        
        pred_weights = torch.sigmoid(pred_weights_bias[:, :self.prediction_length])
        pred_weights = torch.clamp(pred_weights, min=0, max=1)
        pred_biases = F.leaky_relu(pred_weights_bias[:, self.prediction_length:])

        return {
            "weights": weights,  # 表示5个输入序列中，最后一个GNSS观测的权重
            "biases": biases,    # 表示5个输入序列中，最后一个GNSS观测的偏置
            "pred_weights": pred_weights.transpose(0, 1),  # 表示未来8个时间步的权重
            "pred_biases": pred_biases.transpose(0, 1),    # 表示未来8个时间步的偏置
        }
        


class CCFFM(nn.Module):
    def __init__(self, channels, heads=4):
        super(CCFFM, self).__init__()
        self.channels = channels
        self.heads = heads
        self.qkv_proj = nn.ModuleList([nn.Linear(channels, channels) for _ in range(6)])  # 2x(q,k,v)
        self.out_proj = nn.Linear(2 * channels, channels)

    def forward(self, x1, x2, mask=None):
        # x1, x2: [batch_size, max_satellites, 64]
        # mask: [batch_size, max_satellites]，True表示有效
        B, S, C = x1.size()
        device = x1.device
        # mask处理：无效卫星特征置零
        if mask is not None:
            mask_expand = mask.unsqueeze(-1).expand(-1, -1, C)  # [B, S, C]
            x1 = x1 * mask_expand.float()
            x2 = x2 * mask_expand.float()
        # 伪造空间维度以适配原有实现
        x1_reshape = x1.unsqueeze(-1).unsqueeze(-1)  # [B, S, C, 1, 1]
        x2_reshape = x2.unsqueeze(-1).unsqueeze(-1)
        # 合并S维到batch
        x1_flat = x1_reshape.view(B * S, C, 1, 1)
        x2_flat = x2_reshape.view(B * S, C, 1, 1)
        # flatten spatial: B,C,H,W -> B,H*W,C
        def flatten(x):
            return x.flatten(2).transpose(1, 2)
        x1_flatten = flatten(x1_flat)  # [B*S, 1, C]
        x2_flatten = flatten(x2_flat)
        # QKV projection
        q1 = self.qkv_proj[0](x1_flatten)
        k1 = self.qkv_proj[1](x1_flatten)
        v1 = self.qkv_proj[2](x1_flatten)
        q2 = self.qkv_proj[3](x2_flatten)
        k2 = self.qkv_proj[4](x2_flatten)
        v2 = self.qkv_proj[5](x2_flatten)
        def cross_attn(q, k, v):
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.channels**0.5)
            attn = F.softmax(scores, dim=-1)
            return torch.matmul(attn, v)
        out1 = (cross_attn(q1, k2, v2) + cross_attn(q1, k1, v1)) / 2
        out2 = (cross_attn(q2, k1, v1) + cross_attn(q2, k2, v2)) / 2
        out_cat = torch.cat([out1, out2], dim=-1)  # [B*S, 1, 2C]
        out = self.out_proj(out_cat)  # [B*S, 1, C]
        out = out.transpose(1, 2).view(B, S, C)
        # 输出也按mask置零
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out
