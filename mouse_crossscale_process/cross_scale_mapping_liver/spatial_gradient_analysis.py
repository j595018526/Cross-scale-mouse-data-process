#############################################根据指标数值绘制MRI+病理+基因表达变化曲线#####################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import make_interp_spline

# ================== 用户配置 ==================
file_path = "tumor_8_liver_sliding_features.xlsx"
cell_name = "Macrophage_M1"
gene_name = "mm10___Cd74"
corr_method = "spearman"  # 可选 "pearson" 或 "spearman"
smooth = True  # 是否启用平滑
smooth_points = 200  # 平滑后的点数

# 垂直虚线配置：设置要标记的x轴位置（可以一个或多个值）
vertical_lines = [14]
# =============================================

# 读取三个工作表
df_expr = pd.read_excel(file_path, sheet_name="total_expression")
df_cell = pd.read_excel(file_path, sheet_name="cell_count")
df_mri = pd.read_excel(file_path, sheet_name="MRI")

# 提取数据（按 ID 对齐）
ids = df_expr["ID"].values
gene_data = df_expr[gene_name].values.astype(float)
cell_data = df_cell[cell_name].values.astype(float)
mri_data = df_mri["MRI_std"].values.astype(float)

# 归一化到 [0,1]（Min-Max）
gene_norm = (gene_data - np.min(gene_data)) / (np.max(gene_data) - np.min(gene_data))
cell_norm = (cell_data - np.min(cell_data)) / (np.max(cell_data) - np.min(cell_data))
mri_norm = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))

# 计算相关系数
if corr_method == "pearson":
    corr_mri_cell, pval_mri_cell = pearsonr(mri_norm, cell_norm)
    corr_mri_gene, pval_mri_gene = pearsonr(mri_norm, gene_norm)
else:
    corr_mri_cell, pval_mri_cell = spearmanr(mri_norm, cell_norm)
    corr_mri_gene, pval_mri_gene = spearmanr(mri_norm, gene_norm)

print(f"{corr_method.capitalize()} 相关系数 MRI vs {cell_name}: {corr_mri_cell:.2f}, p-value: {pval_mri_cell:.2f}")
print(f"{corr_method.capitalize()} 相关系数 MRI vs {gene_name}: {corr_mri_gene:.2f}, p-value: {pval_mri_gene:.2f}")

# ========== Cell 子刊风格配色 ==========
color_gene = "#2ca02c"   # 翠绿
color_cell = "#2c3e7a"  # 深靛蓝
color_mri = "#d62728"  # 经典红

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

plot_ids, plot_gene = smooth_data(ids, gene_norm, smooth_points, ids)
_, plot_cell = smooth_data(ids, cell_norm, smooth_points, ids)
_, plot_mri = smooth_data(ids, mri_norm, smooth_points, ids)

# ========== 绘图 ==========
plt.figure(figsize=(10, 5), facecolor="white")
plt.plot(plot_ids, plot_gene, linestyle='-', linewidth=2.5, color=color_gene, label=f'Gene: {gene_name}')
plt.plot(plot_ids, plot_cell, linestyle='-', linewidth=2.5, color=color_cell, label=f'Cell: {cell_name}')
plt.plot(plot_ids, plot_mri, linestyle='-', linewidth=2.5, color=color_mri, label='MRI_std')

# 绘制垂直虚线（根据 vertical_lines 列表）
for x_pos in vertical_lines:
    plt.axvline(x=x_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

plt.xlabel("No. of sliding Windows", fontsize=12, fontweight='medium')
plt.ylabel("Normalized value", fontsize=12, fontweight='medium')
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
legend = plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11, edgecolor='lightgray')
legend.get_frame().set_alpha(0.85)
plt.xticks(ids[::2])
plt.tight_layout()
plt.show()

# ========== 打印数据统计信息 ==========
print(f"\n数据统计:")
print(f"基因 {gene_name}: 范围 [{gene_norm.min():.4f}, {gene_norm.max():.4f}], 均值 {gene_norm.mean():.4f}")
print(f"细胞 {cell_name}: 范围 [{cell_norm.min():.4f}, {cell_norm.max():.4f}], 均值 {cell_norm.mean():.4f}")
print(f"MRI_std: 范围 [{mri_norm.min():.4f}, {mri_norm.max():.4f}], 均值 {mri_norm.mean():.4f}")
print(f"数据点数: {len(ids)}")
if smooth:
    print(f"平滑后点数: {len(plot_ids)}")
print(f"垂直虚线位置: {vertical_lines if vertical_lines else '无'}")
#############################################根据指标数值绘制MRI+病理+基因表达变化曲线#####################################
