import torch
from torch.utils.data import DataLoader
from dataset import GNSSDataset
from model import GNSS_Transformer
import numpy as np
import matplotlib.pyplot as plt

# 配置
DATA_DIR = 'data'
TIME_SERIES = {(1623296154, 1623296357), (1623296918, 1623297126)}
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'pths/gnss_transformer_epoch50.pth'  # 修改为实际模型路径

# 数据集
predict_dataset = GNSSDataset(data_dir=DATA_DIR, time_series=TIME_SERIES)
predict_loader = DataLoader(predict_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型参数
max_sats = 40
feature_dim = 4 * max_sats
model = GNSS_Transformer(feature_dim=feature_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


all_pred_traj = []
all_current_pos = []
all_gt_traj = []  # 真值轨迹（后5帧）
all_gt_pos = []   # 当前帧真值

with torch.no_grad():
    for xs, ys in predict_loader:
        xs = xs.to(DEVICE)
        ys = ys.to(DEVICE)
        pred_traj, current_pos = model(xs)
        all_pred_traj.append(pred_traj.cpu().numpy())
        all_current_pos.append(current_pos.cpu().numpy())
        # 保存真值
        all_gt_traj.append(ys[:, 3:, :].cpu().numpy())  # [B, 5, 3]
        all_gt_pos.append(ys[:, 3, :].cpu().numpy())    # [B, 3]


all_pred_traj = np.concatenate(all_pred_traj, axis=0)  # [N, 5, 3]
all_current_pos = np.concatenate(all_current_pos, axis=0)  # [N, 3]
all_gt_traj = np.concatenate(all_gt_traj, axis=0)      # [N, 5, 3]
all_gt_pos = np.concatenate(all_gt_pos, axis=0)        # [N, 3]


print('预测轨迹 shape:', all_pred_traj.shape)
print('当前定位 shape:', all_current_pos.shape)
print('示例 预测轨迹:', all_pred_traj[0])
print('示例 当前定位:', all_current_pos[0])

# 画出高精度定位轨迹与真值对比
plt.figure(figsize=(8, 6))
plt.plot(all_gt_pos[:, 0], all_gt_pos[:, 1], label='GT Traj', c='g')
plt.plot(all_current_pos[:, 0], all_current_pos[:, 1], label='Pred Traj', c='r', alpha=0.7)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Current Position Trajectory')
plt.legend()
plt.savefig('current_pos_traj.png')
plt.close()

# 轨迹预测误差（后5帧）
traj_err = np.linalg.norm(all_pred_traj - all_gt_traj, axis=2)  # [N, 5]
mean_traj_err = traj_err.mean()
print(f'轨迹预测平均误差: {mean_traj_err:.4f}')

# 当前定位误差
cur_err = np.linalg.norm(all_current_pos - all_gt_pos, axis=1)
mean_cur_err = cur_err.mean()
print(f'当前定位平均误差: {mean_cur_err:.4f}')

# 可选：保存结果
np.save('pred_traj.npy', all_pred_traj)
np.save('current_pos.npy', all_current_pos)
np.save('gt_traj.npy', all_gt_traj)
np.save('gt_pos.npy', all_gt_pos)
