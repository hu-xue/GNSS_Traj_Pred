import os
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import rtk_util as util


class GNSSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.obs_files = sorted(glob.glob(os.path.join(data_dir, "*.obs")))
        self.gt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        self.sta_dir = os.path.join(data_dir, "sta")
        self.sta_files = sorted(glob.glob(os.path.join(self.sta_dir, "*")))
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        for obs_path, gt_path in zip(self.obs_files, self.gt_files):
            # 读取观测数据
            obs, nav, sta = util.read_obs(obs_path, self.sta_files)
            obss = util.split_obs(obs)
            # 读取真值
            gt = pd.read_csv(
                gt_path,
                header=None,
                skiprows=30,
                engine="python",
                sep=" +",
                skipfooter=4,
            )  # 跳过前30行注释
            # 每个观测样本与真值匹配
            for o in obss:
                t = o.data[0].time
                t = t.time + t.sec
                gt_row = gt.loc[(gt[0] + 18 - t).abs().argmin()]
                label = [
                    gt_row[3] + gt_row[4] / 60 + gt_row[5] / 3600,
                    gt_row[6] + gt_row[7] / 60 + gt_row[8] / 3600,
                    gt_row[9],
                ]
                self.samples.append((o, nav, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        o, nav, label = self.samples[idx]
        ret = util.get_ls_pnt_pos(o, nav)
        max_sats = 40  # 最大卫星数，可根据实际情况调整
        feature_dim = 4  # SNR, azel(0), azel(1), resd
        if not ret["status"]:
            x = np.zeros(max_sats * feature_dim, dtype=np.float32)
        else:
            exclude = ret["data"]["exclude"]
            SNR = np.array(ret["data"]["SNR"])
            azel = np.delete(
                np.array(ret["data"]["azel"]).reshape((-1, 2)), exclude, axis=0
            )
            resd = np.array(ret["data"]["residual"])
            n = min(len(SNR), max_sats)
            features = np.zeros((max_sats, feature_dim), dtype=np.float32)
            for i in range(n):
                features[i, 0] = SNR[i]
                features[i, 1] = azel[i, 0] if azel.shape[1] > 0 else 0
                features[i, 2] = azel[i, 1] if azel.shape[1] > 1 else 0
                features[i, 3] = resd[i]
            x = features.flatten()
        if self.transform:
            x = self.transform(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


# 用法示例
if __name__ == "__main__":
    dataset = GNSSDataset(data_dir="data")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print("loader len:", len(loader))
    for x, y in loader:
        print(x.shape, y.shape)
