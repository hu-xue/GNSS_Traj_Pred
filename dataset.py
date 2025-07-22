import os
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import rtk_util as util


class GNSSDataset(Dataset):
    def __init__(self, data_dir, transform=None, time_series=None, seq_len=8):
        self.data_dir = data_dir
        self.transform = transform
        self.time_series = time_series if time_series is not None else set()
        self.obs_files = sorted(glob.glob(os.path.join(data_dir, "*.obs")))
        self.gt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        self.sta_dir = os.path.join(data_dir, "sta")
        self.sta_files = sorted(glob.glob(os.path.join(self.sta_dir, "*")))
        self.sequences = []  # 每个time_series区间内的样本list
        self.seq_index = []  # (区间idx, 滑窗起点idx)
        self.seq_len = seq_len  # 每个序列长度
        self._prepare_sequences()

    def _prepare_sequences(self):
        # 对每个time_series区间，收集属于该区间的样本，形成子序列
        for obs_path, gt_path in zip(self.obs_files, self.gt_files):
            obs, nav, sta = util.read_obs(obs_path, self.sta_files)
            obss = util.split_obs(obs)
            gt = pd.read_csv(
                gt_path,
                header=None,
                skiprows=30,
                engine="python",
                sep=" +",
                skipfooter=4,
            )
            # 对每个区间分别收集
            for start, end in self.time_series:
                seq = []
                for o in obss:
                    t = o.data[0].time
                    t = t.time + t.sec
                    if not (start <= t <= end):
                        continue
                    gt_row = gt.loc[(gt[0] + 18 - t).abs().argmin()]
                    label = [
                        gt_row[3] + gt_row[4] / 60 + gt_row[5] / 3600,
                        gt_row[6] + gt_row[7] / 60 + gt_row[8] / 3600,
                        gt_row[9],
                    ]
                    seq.append((o, nav, label))
                if len(seq) >= self.seq_len:
                    self.sequences.append(seq)
        # 记录所有可用滑窗序列的索引
        self.seq_index = []
        for seq_idx, seq in enumerate(self.sequences):
            for i in range(len(seq) - self.seq_len + 1):
                self.seq_index.append((seq_idx, i))

    def __len__(self):
        return len(self.seq_index)

    def __getitem__(self, idx):
        seq_idx, start = self.seq_index[idx]
        seq = self.sequences[seq_idx][start : start + self.seq_len]
        max_sats = 40
        feature_dim = 4
        xs = []
        ys = []
        gpss = []
        os = []
        navs = []
        for o, nav, label in seq:
            ret = util.get_ls_pnt_pos(o, nav)
            if not ret["status"]:
                x = np.zeros(max_sats * feature_dim + 3, dtype=np.float32)
                gps = np.zeros(3, dtype=np.float32)
            else:
                exclude = ret["data"]["exclude"]
                gps = ret["pos"][:3]
                SNR = np.array(ret["data"]["SNR"])
                azel = np.delete(
                    np.array(ret["data"]["azel"]).reshape((-1, 2)), exclude, axis=0
                )
                resd = np.array(ret["data"]["residual"])
                n = min(len(SNR), max_sats)
                indata = np.zeros((max_sats, 3), dtype=np.float32)
                valid = np.hstack(
                    [SNR.reshape(-1, 1), azel[:, 1:], resd.reshape(-1, 1)]
                )
                indata[:n, :] = valid
                x = indata
            if self.transform:
                x = self.transform(x)
            gpss.append(gps)
            xs.append(x)
            ys.append(label)
            os.append(o)
            navs.append(nav)
        xs = torch.from_numpy(np.stack(xs)).float()  # [seq_len, feature_dim, channel]
        gpss = torch.from_numpy(np.stack(gpss)).float()  # [seq_len, gps_dim]
        ys = torch.from_numpy(np.stack(ys)).float()  # [seq_len, label_dim]
        return xs, gpss, ys, os, navs


# 用法示例
if __name__ == "__main__":
    time_series = {
        (1623297151, 1623297556),
    }  # 读取klt3
    dataset = GNSSDataset(data_dir="data", time_series=time_series)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print("loader len:", len(loader))
    for x, gps, y in loader:
        print(x.shape, gps.shape, y.shape)
