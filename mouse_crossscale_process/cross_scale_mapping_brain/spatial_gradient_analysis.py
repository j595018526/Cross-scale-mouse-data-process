# 根据表格指标计算相关性，并绘制跨尺度耦合变化曲线
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import spearmanr

# ================== 用户配置 ==================
file_path = "./brain_data/Sliding_features_m3.xlsx"
gene_name = "Dlg4"
smooth = True  # 是否启用平滑
smooth_points = 200  # 平滑后的点数

# 垂直虚线配置：设置要标记的x轴位置（可以一个或多个值）
vertical_lines = [9, 16, 36]
# =============================================

# 读取工作表（只读取box_group1）
df = pd.read_excel(file_path, sheet_name="box_group1")

# 提取数据：ID作为横坐标，cell列作为细胞数据，gene_name列作为基因数据
ids = df["ID"].values
cell_data = df["cell"].values.astype(float)
gene_data = df[gene_name].values.astype(float)

# 提取T2和ADC数据
t2_data = df["T2"].values.astype(float)
# adc_data = df["ADC"].values.astype(float)

# ========== 计算Spearman相关系数并立即打印结果 ==========
print(f"\n{'='*60}")
print(f"Spearman相关系数分析")
print(f"{'='*60}")

# T2 vs Cell
corr_t2_cell, p_t2_cell = spearmanr(t2_data, cell_data)
print(f"\nT2 vs Cell:")
print(f"  r = {corr_t2_cell:.2f}, p = {p_t2_cell:.4f}")

# # ADC vs Cell
# corr_adc_cell, p_adc_cell = spearmanr(adc_data, cell_data)
# print(f"\nADC vs Cell:")
# print(f"  r = {corr_adc_cell:.2f}, p = {p_adc_cell:.4f}")

# Cell vs Gene
corr_cell_gene, p_cell_gene = spearmanr(cell_data, gene_data)
print(f"\nCell vs {gene_name}:")
print(f"  r = {corr_cell_gene:.2f}, p = {p_cell_gene:.4f}")

# T2 vs Gene
corr_t2_gene, p_t2_gene = spearmanr(t2_data, gene_data)
print(f"\nT2 vs {gene_name}:")
print(f"  r = {corr_t2_gene:.2f}, p = {p_t2_gene:.4f}")

# # ADC vs Gene
# corr_adc_gene, p_adc_gene = spearmanr(adc_data, gene_data)
# print(f"\nADC vs {gene_name}:")
# print(f"  r = {corr_adc_gene:.2f}, p = {p_adc_gene:.4f}")


# ========== 归一化到 [0,1]（Min-Max） ==========
cell_norm = (cell_data - np.min(cell_data)) / (np.max(cell_data) - np.min(cell_data))
gene_norm = (gene_data - np.min(gene_data)) / (np.max(gene_data) - np.min(gene_data))
t2_norm = (t2_data - np.min(t2_data)) / (np.max(t2_data) - np.min(t2_data))
# adc_norm = (adc_data - np.min(adc_data)) / (np.max(adc_data) - np.min(adc_data))

# ========== Cell 子刊风格配色 ==========
# 红色系：T2和ADC使用不同深浅的红色
color_t2 = "#d62728"      # 经典红
color_cell = "#f0b800"    # 金黄色（黄色系）
color_gene = "#2ca02c"    # 翠绿

# ========== 数据平滑处理 ==========
def smooth_data(x, y, smooth_points, ids):
    if smooth and len(ids) > 3:
        x_smooth = np.linspace(ids.min(), ids.max(), smooth_points)
        spline = make_interp_spline(ids, y, k=3)
        y_smooth = spline(x_smooth)
        y_smooth = np.clip(y_smooth, 0, 1)
        return x_smooth, y_smooth
    else:
        return ids, y

plot_ids, plot_cell = smooth_data(ids, cell_norm, smooth_points, ids)
_, plot_gene = smooth_data(ids, gene_norm, smooth_points, ids)
_, plot_t2 = smooth_data(ids, t2_norm, smooth_points, ids)
# _, plot_adc = smooth_data(ids, adc_norm, smooth_points, ids)


# ========== 绘图（绘图时反转数据） ==========
plt.figure(figsize=(10, 5), facecolor="white")

# 绘制三条曲线
plt.plot(plot_ids, plot_t2[::-1], linewidth=2.5, color=color_t2, label='T2 value', alpha=0.85)
plt.plot(plot_ids, plot_cell[::-1], linewidth=2.5, color=color_cell, label='All cells')
plt.plot(plot_ids, plot_gene[::-1], linewidth=2.5, color=color_gene, label=gene_name)

# 绘制垂直虚线
for x_pos in vertical_lines:
    plt.axvline(x=x_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

plt.xlabel("No. of sliding Windows", fontsize=12, fontweight='medium')
plt.ylabel("Normalized value", fontsize=12, fontweight='medium')
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)

# 图例设置
legend = plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11, edgecolor='lightgray')
legend.get_frame().set_alpha(0.85)
plt.xticks(ids[::2])
plt.tight_layout()
plt.show()





# ########################### 根据表格指标绘制跨尺度耦合cell——gene变化曲线 #############################################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import make_interp_spline
#
# # 配置
# file_path = "./brain_data/Sliding_features_b1.xlsx"
# gene_name = "Dlg4"
# vertical_lines = [4, 10, 27]
#
# # 读取并归一化
# df = pd.read_excel(file_path, sheet_name="box_group2")
# ids = df["ID"].values
# cell_norm = (df["cell"] - df["cell"].min()) / (df["cell"].max() - df["cell"].min())
# gene_norm = (df[gene_name] - df[gene_name].min()) / (df[gene_name].max() - df[gene_name].min())
#
# # 平滑
# x_smooth = np.linspace(ids.min(), ids.max(), 200)
# cell_smooth = np.clip(make_interp_spline(ids, cell_norm, k=3)(x_smooth), 0, 1)
# gene_smooth = np.clip(make_interp_spline(ids, gene_norm, k=3)(x_smooth), 0, 1)
#
# # 绘图
# plt.figure(figsize=(10, 5))
# plt.plot(x_smooth, cell_smooth[::-1], linewidth=2.5, color="#f0b800", label='All cells')
# plt.plot(x_smooth, gene_smooth[::-1], linewidth=2.5, color="#2ca02c", label=gene_name)
# for x in vertical_lines:
#     plt.axvline(x=x, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
# plt.xlabel("No. of sliding Windows")
# plt.ylabel("Normalized value")
# plt.ylim(-0.05, 1.05)
# plt.grid(True, linestyle='--', alpha=0.4)
# plt.legend(loc='best', frameon=True, fancybox=True)
# plt.xticks(ids[::2])
# plt.tight_layout()
# plt.show()
# ########################### 根据表格指标绘制跨尺度耦合cell——gene变化曲线 #############################################
