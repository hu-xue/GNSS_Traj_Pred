import numpy as np
import matplotlib.pyplot as plt
import sys
import pymap3d as p3d

# import pandas as pd

try:
    path = sys.argv[1]
    tdl_file = sys.argv[2]
    vgp_file = sys.argv[3]
except Exception:
    print(Exception)
    # klt1_predict klt2_predict urban_medium urban_harsh
    path = "result/klt2"
    tdl_file = "tdl"
    vgp_file = "vgp"
    
print("path: ", path, "tdl_file: ", tdl_file, "vgp_file: ", vgp_file)

try:
    title = sys.argv[4] + " "
except Exception:
    print(Exception)
    title = ""

item = ["gt", "go", "rtk", "TDL_GNSS", "VGP"]

labels = ["goGPS", "RTKLIB", "TDL", "Ground Truth", "VGP"]
colors = ["orange", "brown", "green", "red", "blue"]

labels1 = ["goGPS", "RTKLIB", "TDL", "VGP"]
# colors1 = ["orange", "brown", "red", "blue"]

gtpos = np.loadtxt(path + "/gt.csv", skiprows=1, delimiter=",")
gopos = np.loadtxt(path + "/gogps_pos.csv", skiprows=1, delimiter=",")
rtkpos = np.loadtxt(path + "/rtklib_pos.csv", skiprows=1, delimiter=",")
# bpos = np.loadtxt(path+"/TDL_bias_pos.csv",skiprows=1,delimiter=',')
# wpos = np.loadtxt(path+"/TDL_weight_pos.csv",skiprows=1,delimiter=',')
TDL_pos = np.loadtxt(path + f"/{tdl_file}.csv", skiprows=1, delimiter=",")
VGP_pos = np.loadtxt(path + f"/{vgp_file}.csv", skiprows=1, delimiter=",")

print('len gt:', len(gtpos), 'len go:', len(gopos), 'len rtk:', len(rtkpos), 'len TDL:', len(TDL_pos), 'len VGP:', len(VGP_pos))

gt_init = gtpos[0]

errors = []
traj = []

for gt, go, rtk, tdl, vgp in zip(gtpos, gopos, rtkpos, TDL_pos, VGP_pos):
    errors.append(
        [
            p3d.geodetic2enu(*go, *gt),
            p3d.geodetic2enu(*rtk, *gt),
            p3d.geodetic2enu(*tdl, *gt),
            p3d.geodetic2enu(*vgp, *gt),
        ]
    )
    traj.append(
        [
            p3d.geodetic2enu(*go, *gt_init),
            p3d.geodetic2enu(*rtk, *gt_init),
            p3d.geodetic2enu(*tdl, *gt_init),
            p3d.geodetic2enu(*gt, *gt_init),
            p3d.geodetic2enu(*vgp, *gt_init),
        ]
    )

errors = np.array(errors)
traj = np.array(traj)

print("Methods: ", labels1)
print("2D error: ", np.linalg.norm(errors[:, :, :2], axis=2).mean(axis=0))
print("3D error: ", np.linalg.norm(errors, axis=2).mean(axis=0))

plt.figure(figsize=(6, 6))
for i in range(5):
    plt.plot(
        traj[:, i, 0],
        traj[:, i, 1],
        marker="o",
        linestyle="-",
        color=colors[i],
        label=labels[i],
        markersize=3,
    )
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 1, 2, 3, 4]
plt.legend(
    [handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right"
)
plt.title(title + "2D Trajectory(m)")
plt.axis("equal")
plt.tight_layout()
plt.savefig(path + f"/{path.split('/')[-1]}_traj2d.eps", format="eps")
plt.savefig(path + f"/{path.split('/')[-1]}_traj2d.pdf", format="pdf")
# plt.show()

# copy path + f"/{path.split('/')[-1]}_traj2d.eps" to ../paper/v0.1/img
# import shutil
# shutil.copy( path + f"/{path.split('/')[-1]}_traj2d.eps", "../paper/v0.1/img")

# "font.size": 30,
# plt.rcParams.update({"font.family": "Times New Roman"})

# errors3d = {}

# plt.figure(figsize=(16, 10))
# for i in range(5):
#     plt.plot(np.linalg.norm(errors[:,i,:],axis=1),label = labels1[i], color=colors1[i])
#     errors3d[labels1[i]] = np.linalg.norm(errors[:,i,:],axis=1)
# plt.xlim(left=0)
# plt.legend()
# plt.title(title+"3D Error(m)")
# plt.ylabel("Error(m)")
# plt.tight_layout()
# plt.savefig(path+"/error3d.png")

# fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 10))

# 处理每个标签和对应的颜色
# errors3d = {}
# for i in range(3):
#     norm_data = np.linalg.norm(errors[:, i, :], axis=1)
#     errors3d[labels1[i]] = norm_data
#     # 绘制到子图
#     ax.plot(norm_data, label=labels1[i], color=colors1[i])
#     ax2.plot(norm_data, label=labels1[i], color=colors1[i])

# # 设置Y轴范围
# ax.set_ylim(200, 1200)  # 上图显示较大的错误值
# ax2.set_ylim(0, 200)  # 下图显示较小的错误值

# 隐藏两个子图间的间隙
# ax.spines["bottom"].set_visible(False)
# ax2.spines["top"].set_visible(False)
# ax.xaxis.tick_top()
# ax.tick_params(labeltop=False)
# ax2.xaxis.tick_bottom()

# 断裂标记参数
# d = 0.015  # 斜杠大小
# kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
# kwargs.update(transform=ax2.transAxes)
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# 其他设置
# ax.set_title(title + " 3D Error (m)")
# ax.set_ylabel("Error (m)")
# ax.legend()
# plt.tight_layout()

# 保存图片
# plt.savefig(path + "/error3d.png")
# plt.show()


# errors3d = pd.DataFrame(errors3d)
# errors3d.to_csv(path + "/errors3d.csv", index=None)


# Redraw the boxplot with Times New Roman font
# plt.figure(figsize=(12, 6))
# plt.boxplot(
#     [errors3d["goGPS"], errors3d["RTKLIB"], errors3d["Ours"]],
#     labels=["goGPS", "RTKLIB", "Ours"],
# )
# plt.title(f"Boxplot of 3D Positioning Errors on {title} Dataset")
# plt.ylabel("Error (meters)")
# plt.grid(True)
# plt.savefig(path + "/error3dbox.png")


# plt.figure(figsize=(16, 10))
# for i in range(3):
#     plt.plot(
#         np.linalg.norm(errors[:, i, :2], axis=1), label=labels1[i], color=colors1[i]
#     )
# plt.xlim(left=0)
# plt.legend()
# plt.title(title + "2D Error(m)")
# plt.ylabel("Error(m)")
# plt.tight_layout()
# plt.savefig(path + "/error2d.png")
