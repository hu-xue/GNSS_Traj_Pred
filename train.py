from ast import arg
import re
import torch
import pyrtklib as prl
import rtk_util as util
import json
import sys
import numpy as np
import pandas as pd
import pymap3d as p3d
from model import VGPNet
from torch.nn import MSELoss
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms
import argparse
import datetime

DEVICE = "cuda"

# use argparse to get config
parser = argparse.ArgumentParser(description="Train the model")
parser.add_argument(
    "--config_file",
    type=str,
    default="config/train_img/klt3_train.json",
    help="Path to the config file",
)
parser.add_argument("--bool_gnss", action="store_true", help="Boolean flag for GNSS usage")
parser.add_argument("--bool_fisheye", action="store_true", help="Boolean flag for fisheye usage")
parser.add_argument("--bool_surrounding", action="store_true", help="Boolean flag for surrounding usage")
parser.add_argument("--bool_mask", action="store_true", help="Boolean flag for mask usage")
parser.add_argument("--bool_ccffm", action="store_true", help="Boolean flag for CCFFM usage")
parser.add_argument("--resume", type=int, default=0, help="Resume training from this epoch")
parser.add_argument("--epoch", type=int, default=200, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
args = parser.parse_args()
print("Arguments:", args)
config_file = args.config_file
print("Config file:", config_file)


dataset_name = config_file.split("/")[-1].split(".json")[0].split("_")[0]
print(dataset_name)
# 后缀
if dataset_name in ["klt1", "klt2", "klt3"]:
    ends = "png"
elif dataset_name in ["rw1", "rw2", "rw3"]:
    ends = "jpg"
print("ends: ", ends)
# urban_deep or klt3_train
# config = "config/image/klt3_train.json"

with open(config_file) as f:
    conf = json.load(f)

mode = conf["mode"]
if mode not in ["train", "predict"]:
    raise RuntimeError("%s is not a valid option" % mode)

result = config_file.split("/")[-1].split(".json")[0]


bool_gnss = args.bool_gnss
bool_fisheye = args.bool_fisheye
bool_ccffm = args.bool_ccffm

prefix = "_"
if bool_gnss:
    prefix += "g"
if bool_fisheye:
    prefix += "f"
prefix += "_ccffm" if bool_ccffm else ""

now = datetime.datetime.now()
prefix += f"_{now.strftime('%Y%m%d_%H%M%S')}"

print("prefix: ", prefix)

result_path = "result/train_img/" + result + prefix
print(result_path)
model_dir = conf["model"] + prefix
print(model_dir)

os.makedirs(result_path, exist_ok=True)  # dir for result
os.makedirs(model_dir, exist_ok=True)  # dir for model


obs, nav, sta = util.read_obs(conf["obs"], conf["eph"])
prl.sortobs(obs)

obss = util.split_obs(obs)
print(f"obs num: {len(obss)}")

tmp = []

#! diff about the dataset
if conf.get("gt", None):
    gt = None
    if dataset_name == "klt3" or dataset_name == "klt1" or dataset_name == "klt2":
        gt = pd.read_csv(
            conf["gt"],
            skiprows=30,
            header=None,
            sep=" +",
            skipfooter=4,
            engine="python",
        )
        gt[0] = gt[0] + 18  # leap seconds
        # time_ref = pd.read_csv(conf["ref"], header=0)
    elif dataset_name == "rw1" or dataset_name == "rw2":
        gt = pd.read_csv(conf["gt"], header=None)
        gt[6] = gt[6] + 18  # leap seconds
        # read ros time ref
        # time_ref = pd.read_csv(conf["ref"], header=0)
    gts = []

# load image infos
img_fishs = []
if conf.get("img", None) and (bool_fisheye):
    img_fish = os.listdir(os.path.join(conf["img"], "fisheye"))
    for i in img_fish:
        if i.endswith(".png") or i.endswith(".jpg"):
            img_fishs.append(float(i[:-4]))
    img_fishs.sort()

# filter and normalize
gather_data = []
# obs_index = 0
for o in obss:
    t = o.data[0].time
    t = t.time + t.sec
    if t > conf["start_time"] and (conf["end_time"] == -1 and 1 or t < conf["end_time"]):
        tmp.append(o)
        if conf.get("gt", None):
            gt_row = (
                gt.loc[(gt[0] - t).abs().argmin()]
                if (dataset_name not in ["rw1", "rw2", "rw3"])
                else gt.loc[(gt[6] - t).abs().argmin()]
            )
            gt_time = gt_row[0] if dataset_name not in ["rw1", "rw2", "rw3"] else gt_row[6]
            # print("gps time: ", t)
            # print("gt-time : ", gt_time)
            if dataset_name in ["rw1", "rw2", "rw3"]:
                gts.append([gt_row[1], gt_row[2], gt_row[3]])
            else:
                gts.append(
                    [
                        gt_row[3] + gt_row[4] / 60 + gt_row[5] / 3600,
                        gt_row[6] + gt_row[7] / 60 + gt_row[8] / 3600,
                        gt_row[9],
                    ]
                )

        # if conf.get("img", None) and bool_fisheye:
        #     if dataset_name in ["rw1", "rw2", "rw3", "klt1", "klt2", "klt3"]:
        #         # 在time_ref根据t找到对应的t1
        #         time_tmp = (time_ref["time_ref"] - (gt_time - 18)).abs().idxmin()
        #         ros_img_time = time_ref.loc[time_tmp, "ros_time"]
        #         ref_gt_time = time_ref.loc[time_tmp, "time_ref"]
        #     else:
        #         ros_img_time = gt_time - 18
        #         ref_gt_time = gt_time - 18
                
        #     print("imt-time: ", ros_img_time)
        #     img_fish_row = min(img_fishs, key=lambda x: abs(x - ros_img_time))
        #     img_row_f = os.path.join(conf["img"], "fisheye", f"{img_fish_row:.6f}.{ends}")
        #     img_fishes.append(img_row_f)

        ret = util.get_ls_pnt_pos(o, nav)  # 计算最小二乘解
        if not ret["status"]:
            continue
        rs = ret["data"]["eph"]
        dts = ret["data"]["dts"]
        sats = ret["data"]["sats"]
        exclude = ret["data"]["exclude"]
        prs = ret["data"]["prs"]
        resd = np.array(ret["data"]["residual"])
        SNR = np.array(ret["data"]["SNR"])
        azel = np.delete(np.array(ret["data"]["azel"]).reshape((-1, 2)), exclude, axis=0)
        gather_data.append(np.hstack([SNR.reshape(-1, 1), azel[:, 1:], resd]))
        # print(f"obs time: {t}")
        # file_name_txt = f"{t:.3f}.{ends}"
        # row_file_dir = os.path.join(conf["img"], "fisheye",f"{img_fishs[obs_index]:6f}.png")
        # dist_file_dir = os.path.join(conf["img"], "fisheye",file_name_txt)
        # # rename
        # os.rename(row_file_dir, dist_file_dir)
        # print(f"rename {row_file_dir} to {dist_file_dir}")
        
        # obs_index += 1
        # 给图片对应序号的图片重命名为t
        
        
# print(f"total obs num: {len(tmp)}")

print(f"gather data num: {len(gather_data)}")
norm_data = np.vstack(gather_data)
imean = norm_data.mean(axis=0)
istd = norm_data.std(axis=0)

print(f"preprocess done, mean:{imean}, std:{istd}")


net = VGPNet(
    torch.tensor(imean, dtype=torch.float32),
    torch.tensor(istd, dtype=torch.float32),
    sequence_length=3,     # 输入序列长度
    prediction_length=2,   # 预测序列长度
)
net = net.double().to(DEVICE)

resume_ep = args.resume
model_path = model_dir + f"/image_ep{resume_ep}.pth"
if os.path.exists(model_path) and resume_ep > 0:
    net.load_state_dict(torch.load(model_path))
    print(f"load from {model_path}.")
else:
    print(f"model path {model_path} not found or not config resume, starting from scratch.")
    resume_ep = 0


obss = tmp

pos_errs = []


opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

epoch = args.epoch

loss_log_path = result_path + f"/loss_{resume_ep}.csv"
if os.path.exists(loss_log_path):
    vis_loss = list(np.loadtxt(loss_log_path).reshape(-1))
else:
    vis_loss = []


f_preprocess = (
    transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=conf["f_norm_std"][0], std=conf["f_norm_std"][1])]
    )
    if bool_fisheye
    else None
)


for k in range(resume_ep, epoch):
    # 修改为序列长度为5的输入，预测长度为8的输出
    sequence_length = 3   # 输入历史序列长度
    prediction_length = 2  # 预测未来长度
    
    # 创建序列索引，确保有足够的历史数据和未来数据
    valid_indices = list(range(sequence_length - 1, len(obss) - prediction_length))
    # random.shuffle(valid_indices)  # 随机打乱数据
    epoch_num = 0
    epoch_loss = []  # 存储每个epoch的loss
    with tqdm(valid_indices, desc=f"Epoch {k+1}", ncols=80) as t:
        for center_i in t:  # 遍历所有的有效序列中心点
            epoch_num += 1
            batch_loss = 0  # 计算每个batch的loss
            # 构建输入序列：5个历史观测
            input_indices = list(range(center_i - sequence_length + 1, center_i + 1))
            # 构建预测目标：8个未来观测
            target_indices = list(range(center_i + 1, center_i + prediction_length + 1))
            
            # 收集序列数据
            sequence_data = []
            sequence_images = []
            gnss_gts = gts[input_indices[-1]]
            
            # 处理输入序列（5个历史点）
            for i in input_indices:
                o = obss[i]
                ret = util.get_ls_pnt_pos(o, nav)
                
                resd = np.array(ret["data"]["residual"])
                SNR = np.array(ret["data"]["SNR"])
                exclude = ret["data"]["exclude"]
                azel = np.delete(np.array(ret["data"]["azel"]).reshape((-1, 2)), exclude, axis=0)
                
                # 构建输入特征 [SNR, elevation, residual]
                in_data = torch.tensor(np.hstack([SNR.reshape(-1, 1), azel[:, 1:], resd]), dtype=torch.float64).to(DEVICE)
                sequence_data.append(in_data)
                
                # 处理图像
                img_row_f = os.path.join(conf["img"], "fisheye", f"{img_fishs[i]}.{ends}")
                img_f = Image.open(img_row_f).resize((224, 224))
                img_f = f_preprocess(img_f).unsqueeze(0).to(DEVICE, dtype=torch.float64)  
                sequence_images.append(img_f)
            
            # 收集轨迹预测的8个点真值
            target_gts = []
            
            for i in target_indices:
                o = obss[i]
                if conf.get("gt", None):
                    gt_row = gts[i]
                    target_gts.append(gt_row)
            
            # 把5个历史数据输入网络 
            output = net(sequence_data, sequence_images)
            
            weight = output["weights"]
            bias = output["biases"]
            weight_pred = output["pred_weights"]
            bias_pred = output["pred_biases"]
            
            # 用网络预测的权重和偏置来计算最小二乘解
            pred_obss = obss[input_indices[-1]]
            ret_ori = util.get_ls_pnt_pos(pred_obss, nav)
            ret_ori_sats = ret_ori["data"]["sats"]
            ret_ori_exclude = ret_ori["data"]["exclude"]
            sats_used = np.delete(np.array(ret_ori_sats), ret_ori_exclude, axis=0)
            weight = weight[:len(sats_used)]
            bias = bias[:len(sats_used)]
            
            ret = util.get_ls_pnt_pos_torch(pred_obss, nav, torch.diag(weight), bias.reshape(-1, 1), p_init=ret_ori["pos"])
            result_wls = ret["pos"][:3]
            enu = p3d.ecef2enu(*result_wls, gnss_gts[0], gnss_gts[1], gnss_gts[2])
            gnss_positioning_loss = torch.norm(torch.hstack(enu[:3]))  # 每个sample的loss
            
            # 计算预测的x个时间步的位置
            pred_positions = []
            for i, idx in enumerate(target_indices):
                o=pred_obss
                # 计算基于模型的最小二乘解
                ret = util.get_ls_pnt_pos(o, nav)
                ret_sats = ret["data"]["sats"]
                ret_exclude = ret["data"]["exclude"]
                sats_used = np.delete(np.array(ret_sats), ret_exclude, axis=0)
                sats_number = len(sats_used)
                
                ret_pred = util.get_ls_pnt_pos_torch(o, nav, torch.diag(weight_pred[i][:sats_number]), bias_pred[i][:sats_number].reshape(-1, 1), p_init=ret_ori["pos"])
                result_wls_pred = ret_pred["pos"][:3]
                enu_pred = p3d.ecef2enu(*result_wls_pred, target_gts[i][0], target_gts[i][1], target_gts[i][2])
                pred_loss_i = torch.norm(torch.hstack(enu_pred[:3]))
                pred_positions.append(pred_loss_i)
            
            # 计算8个时间步的平均损失
            pred_loss = torch.mean(torch.tensor(pred_positions, dtype=torch.float64, device=DEVICE))
            
            # 计算总损失
            batch_loss += gnss_positioning_loss + pred_loss

            t.set_postfix({"batch_loss": batch_loss.item()})

            # 每4个样本计算一次梯度
            if epoch_num % 24 == 0 and epoch_num > 0:
                batch_loss = batch_loss / 24  # 平均损失
                # 反向传播和优化
                opt.zero_grad()
                batch_loss.backward()
                opt.step()
                epoch_loss.append(batch_loss.item())
                batch_loss = 0
                epoch_num = 0

    vis_loss.append(np.mean(epoch_loss))
    print(f"Epoch {k+1} loss: {vis_loss[-1]}")
    
    
    if k % 10 == 0 and k > 0:
        torch.save(
            net.state_dict(),
            os.path.join(model_dir, f"image_ep{k}.pth"),
        )
        plt.plot(vis_loss)
        plt.savefig(result_path + f"/loss_{k}.png")
        vis_loss_300 = np.array(vis_loss)
        np.savetxt(result_path + f"/loss_{k}.csv", vis_loss_300.reshape(-1, 1))
    sch.step()
        
torch.save(net.state_dict(), os.path.join(model_dir, "image_3d.pth"))
vis_loss = np.array(vis_loss)
plt.plot(vis_loss)
plt.savefig(result_path + f"/loss_{epoch}.png")
np.savetxt(result_path + f"/loss_{epoch}.csv", vis_loss.reshape(-1, 1))
