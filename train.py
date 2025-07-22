import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import GNSSDataset
from model import GNSS_Transformer
import rtk_util as util

# 配置
DATA_DIR = "data"
TIME_SERIES = {(1623297151, 1623297556)}
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集
train_dataset = GNSSDataset(data_dir=DATA_DIR, time_series=TIME_SERIES)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

# 模型参数
max_sats = 40
feature_dim = 3 * max_sats  # 每时刻特征展平为[40,3]
model = GNSS_Transformer(feature_dim=feature_dim).to(DEVICE)

# 损失和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xs, gps, ys in train_loader:
        # xs: [B, 8, 120], gps: [B, 8, 3], ys: [B, 8, 3]
        xs = xs.to(DEVICE)
        gps = gps.to(DEVICE)
        ys = ys.to(DEVICE)
        target_traj = ys[:, 3:, :]  # [B, 5, 3]
        target_pos = ys[:, 3, :]  # [B, 3]
        pred_traj, current_weight, current_bias = model(xs, gps)  # pred_traj: [B, 5, 3], current_weight: [B, 3], current_bias: [B, 3]

        # 可选：用current_weight, current_bias进一步处理

        loss_traj = criterion(pred_traj, target_traj)
        loss_pos = criterion(current_bias, target_pos)  # 以current_bias为当前定位输出
        loss = loss_traj + loss_pos
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xs.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")
    # 可选：每5轮保存一次模型
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"pths/gnss_transformer_epoch{epoch+1}.pth")

print("Training finished.")
