from ast import arg
from sympy import false
import torch
import pyrtklib as prl
import rtk_util as util
import json
import sys
import numpy as np
import pandas as pd
import pymap3d as p3d
from model import VGPNet
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms
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
parser.add_argument("--bool_gnss", action="store_true", help="Boolean flag for GNSS usage", default=False)
parser.add_argument("--bool_fisheye", action="store_true", help="Boolean flag for fisheye usage", default=False)
parser.add_argument("--bool_ccffm", action="store_true", help="Boolean flag for CCFFM usage", default=False)
args = parser.parse_args()
print("Arguments:", args)
config_file = args.config_file
print("Config file:", config_file)

dataset_name = config_file.split("/")[-1].split(".json")[0].split("_")[0]
print(dataset_name)
# 后缀
if "klt" in dataset_name:
    ends = "png"
elif "rw" in dataset_name:
    ends = "jpg"
# klt1_predict klt2_predict urban_medium urban_harsh
# config = "config/image/urban_harsh.json"

with open(config_file) as f:
    conf = json.load(f)

mode = conf["mode"]
if mode not in ["train", "predict"]:
    raise RuntimeError("%s is not a valid option" % mode)

result = config_file.split("/")[-1].split(".json")[0]  # klt1_predict klt2_predict
result = result.split("_")[0]  # klt1 klt2

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

print(prefix)
result_path = "result/pred_image/" + result + prefix
print(result_path)

os.makedirs(result_path, exist_ok=True)
os.makedirs(result_path + "/bw_VGL", exist_ok=True)


net = VGPNet(
    sequence_length=3,     # 输入序列长度
    prediction_length=2,   # 预测序列长度
)
net.double()
net.load_state_dict(torch.load(f"model/image_klt3_gf_ccffm_20250809_033514/image_3d.pth"))
net = net.to(DEVICE)


obs, nav, sta = util.read_obs(conf["obs"], conf["eph"])
prl.sortobs(obs)
prcopt = prl.prcopt_default
obss = util.split_obs(obs)

tmp = []

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
    elif dataset_name == "rw1" or dataset_name == "rw2" or dataset_name == "rw3":
        gt = pd.read_csv(conf["gt"], header=None)
        gt[6] = gt[6] + 18
    gts = []


# load image infos
img_lefts = []
img_rights = []
img_fishs = []
if conf.get("img", None) and (args.bool_fisheye):
    img_fish = os.listdir(os.path.join(conf["img"], "fisheye"))
    for i in img_fish:
        if i.endswith(".png") or i.endswith(".jpg"):
            img_fishs.append(float(i[:-4]))
    img_fishs.sort()


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

f_preprocess = (
    transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=conf["f_norm_std"][0], std=conf["f_norm_std"][1]),
        ]
    )
    if bool_fisheye
    else None
)


obss = tmp
net.eval()
errors = []
gt_pos = []
TDL_bw_pos = []
ecef_pos = []
samples = 0

sequence_length = 3
prediction_length = 2
valid_indices = list(range(sequence_length -1, len(obss) - prediction_length))
with tqdm(valid_indices,desc=f"Predicting...", ncols=80) as t:
    for center_i in t:
        input_indices = list(range(center_i - sequence_length + 1, center_i + 1))
        target_indices = list(range(center_i + 1, center_i + prediction_length + 1))
        
        
        sequence_data = []
        sequence_images = []
        gnss_gts = gts[input_indices[-1]]
        
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
        
        # 保存每个eopch观测到的所有卫星的权重和偏置
        sats_used = np.delete(np.array(ret_ori_sats), ret_ori_exclude, axis=0)
        snp = sats_used
        weight = weight[:len(snp)]
        bias = bias[:len(snp)]
        
        wnp = weight.detach().cpu().numpy()
        bnp = bias.detach().cpu().numpy()
        ep = pd.DataFrame(np.vstack([snp, wnp, bnp]).T)
        ep.columns = ["sat", "weight", "bias"]
        ep.to_csv(result_path + "/bw_VGL/%d.csv" % i, index=None)
        
        # 保存每个eopch的定位结果
        ret = util.get_ls_pnt_pos_torch(pred_obss, nav, torch.diag(weight), bias.reshape(-1, 1), p_init=ret_ori["pos"])
        result_wls = ret["pos"][:3].detach().cpu().numpy()
        TDL_bw_pos.append(p3d.ecef2geodetic(*result_wls))
        gt_pos.append([gnss_gts[0], gnss_gts[1], gnss_gts[2]])
        errors.append(p3d.geodetic2enu(*TDL_bw_pos[-1], *gt_pos[-1]))
        ecef_pos.append(ret["pos"].detach().cpu().numpy())
        
        
        
        
        # 计算预测的8个时间步的位置
        pred_positions_bw_pos = []
        pred_gt_pos = []
        pred_errors = []
        pred_ecef_pos = []
        for i, idx in enumerate(target_indices):
            o=pred_obss

            ret = util.get_ls_pnt_pos(o, nav)
            SNR = np.array(ret["data"]["SNR"])
            sats_number = len(SNR)

            ret_pred = util.get_ls_pnt_pos_torch(o, nav, torch.diag(weight_pred[i][:sats_number]), bias_pred[i][:sats_number].reshape(-1, 1), p_init=ret["pos"])
            result_wls_pred = ret_pred["pos"][:3].detach().cpu().numpy()
            pred_positions_bw_pos.append(p3d.ecef2geodetic(*result_wls_pred))
            pred_gt_pos.append([target_gts[i][0], target_gts[i][1], target_gts[i][2]])
            pred_errors.append(p3d.geodetic2enu(*pred_positions_bw_pos[-1], *pred_gt_pos[-1]))
            pred_ecef_pos.append(ret_pred["pos"].detach().cpu().numpy())

        pred_ecef_pos = np.array(pred_ecef_pos)
        pred_gt_pos = np.array(pred_gt_pos)
        pred_positions_bw_pos = np.array(pred_positions_bw_pos)
        pred_errors = np.array(pred_errors)
        
        np.savetxt(result_path + f"/pred_TPGP{prefix}.csv", pred_positions_bw_pos, delimiter=",", header="lat,lon,height", comments="")
        np.savetxt(result_path + f"/pred_ecef_TPGP{prefix}.csv", pred_ecef_pos, delimiter=",", header="x,y,z,t1,t2,t3,t4", comments="")
        np.savetxt(result_path + f"/pred_gt_TPGP{prefix}.csv", pred_gt_pos, delimiter=",", header="lat,lon,height", comments="")
        np.savetxt(result_path + f"/pred_errors_TPGP{prefix}.csv", pred_errors, delimiter=",", header="east,north,up", comments="")


ecef_pos = np.array(ecef_pos)
gt_pos = np.array(gt_pos)
TDL_bw_pos = np.array(TDL_bw_pos)
errors = np.array(errors)


np.savetxt(result_path + "/ecef_VGP.csv", ecef_pos, delimiter=",", header="x,y,z,t1,t2,t3,t4", comments="")
np.savetxt(result_path + "/gt_VGP.csv", gt_pos, delimiter=",", header="lat,lon,height", comments="")
np.savetxt(result_path + f"/VGP{prefix}.csv", TDL_bw_pos, delimiter=",", header="lat,lon,height", comments="")


print(f"2D mean: {np.linalg.norm(errors[:,:2],axis=1).mean():.2f}, 3D mean: {np.linalg.norm(errors,axis=1).mean():.2f}")
print(f"Samples {samples}")
