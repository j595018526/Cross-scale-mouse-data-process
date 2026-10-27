import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

# ========== 设置中文字体 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def build_heatmap_data(box_size, target_type='count'):
    """
    用滑动窗口构建热图数据
    target_type 支持三种模式:
        'count' - 统计目标细胞类型的数量
        'avg_expression' - 细胞平均基因表达量
        'total_expression' - 细胞总基因表达量
    """
    struct_elem = np.ones((box_size, box_size), dtype=np.uint8)
    valid_start_mask = cv2.erode(tissue_mask, struct_elem)
    valid_starts = np.column_stack(np.where(valid_start_mask == 255))
    step = box_size
    valid_start_positions = [(x, y) for y, x in valid_starts if y % step == 0 and x % step == 0]
    valid_start_positions = sorted(set(valid_start_positions), key=lambda p: (p[1], p[0]))

    boxes_data = []
    for x, y in valid_start_positions:
        in_box = df_cells[(df_cells['X_img'] >= x) & (df_cells['X_img'] < x + box_size) & (df_cells['Y_img'] >= y) & (df_cells['Y_img'] < y + box_size)]

        if target_type == 'count':
            # 统计目标细胞类型数量
            value = len(in_box[in_box['dominant_celltype'] == TARGET_CELL])
        elif target_type == 'avg_expression':
            # 细胞平均基因表达量
            value = in_box[TARGET_GENE].mean() if len(in_box) > 0 else 0
        elif target_type == 'total_expression':
            # 细胞总基因表达量
            value = in_box[TARGET_GENE].sum() if len(in_box) > 0 else 0
        else:
            raise ValueError(f"不支持的 target_type: {target_type}")

        boxes_data.append((x, y, value))

    unique_x = sorted(set(x for x, _, _ in boxes_data))
    unique_y = sorted(set(y for _, y, _ in boxes_data))
    x_to_idx = {x: i for i, x in enumerate(unique_x)}
    y_to_idx = {y: i for i, y in enumerate(unique_y)}
    heatmap = np.full((len(unique_y), len(unique_x)), np.nan)
    for x, y, val in boxes_data:
        heatmap[y_to_idx[y], x_to_idx[x]] = val
    return heatmap, unique_x, unique_y


# ========== 载入路径配置 ==========
base_path = "H:/Cross-scale_mouse_data/ST_data/tumor_8_liver"
he_img_path = f"{base_path}/Liver_8_aligned_HE.png"
mask_path = f"{base_path}/Liver_8_aligned_HE_mask.png"
gene_exp_path = f"{base_path}/tumor_8_liver_target_gene_expression.csv"

cell_locations_path = f"{base_path}/filtered_feature_bc_matrix/cell_locations.tsv"
niche_spot_path = "H:/Cross-scale_mouse_data/ST_data/Niche_spot_coordinates.csv"

# ========== 参数设置 ==========
SAMPLE_NAME = 'Liver_8'
TARGET_CELL = 'HSC_Activated_ECM'
TARGET_GENE = 'mm10___Cd74'
BOX_SIZE_COUNT = 250  # 细胞个数热图的box边长
BOX_SIZE_CD74 = 250  # 基因表达热图的box边长

# ========== 输出路径配置 ==========
output_dir = "D:\PythonCode\mouse_process\image\热图结果\基因总表达"
output_heatmap_count = f"{output_dir}/{SAMPLE_NAME}_{TARGET_CELL}_heatmap.png"
output_heatmap_cd74_avg = f"{output_dir}/{SAMPLE_NAME}_{TARGET_GENE}_avg_heatmap.png"
output_heatmap_cd74_total = f"{output_dir}/{SAMPLE_NAME}_{TARGET_GENE}_total_heatmap.png"

# ========== 加载mask数据 ==========
mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
tissue_mask = cv2.bitwise_not(mask_img)
h, w = tissue_mask.shape
print(f"图像尺寸: {w} x {h}")

df_gene = pd.read_csv(gene_exp_path)
df_cells = pd.read_csv(cell_locations_path, sep='\t')
he_img = np.array(Image.open(he_img_path))
he_h, he_w = he_img.shape[:2]
scale_ratio_x, scale_ratio_y = 56500 / he_w, 20434 / he_h

df_cells['X_img'] = df_cells['X'] / scale_ratio_x
df_cells['Y_img'] = he_h - (df_cells['Y'] / scale_ratio_y)
df_cells = df_cells[(df_cells['X_img'] >= 0) & (df_cells['X_img'] < he_w) & (df_cells['Y_img'] >= 0) & (df_cells['Y_img'] < he_h)]
df_niche = pd.read_csv(niche_spot_path)
df_niche = df_niche[df_niche['sample'] == SAMPLE_NAME].copy()
df_niche = df_niche[['barcode', 'dominant_celltype']].copy()
df_cells = df_cells.merge(df_niche, left_on='Cell_Barcode', right_on='barcode', how='left')
df_cells['dominant_celltype'] = df_cells['dominant_celltype'].fillna('Unknown')

if TARGET_GENE in df_gene.columns:
    gene_exp_dict = dict(zip(df_gene['barcode'], df_gene[TARGET_GENE]))
    df_cells[TARGET_GENE] = df_cells['Cell_Barcode'].map(gene_exp_dict).fillna(0)
else:
    df_cells[TARGET_GENE] = 0

print(f"数据准备完成，共 {len(df_cells)} 个有效细胞")


# ========== 绘制热图1: 细胞数量 ==========
print(f"\n开始分析细胞个数，box尺寸: {BOX_SIZE_COUNT}x{BOX_SIZE_COUNT}")
heatmap_count, x_coords, y_coords = build_heatmap_data(BOX_SIZE_COUNT, 'count')
count_valid = heatmap_count[~np.isnan(heatmap_count)]
count_min, count_max = count_valid.min(), count_valid.max()
heatmap_count_norm = (heatmap_count - count_min) / (count_max - count_min) if count_max > count_min else np.zeros_like(heatmap_count)
print(f"细胞个数 - 原始范围: [{count_min:.2f}, {count_max:.2f}] -> 归一化到 [0, 1]")
print(f"网格尺寸: {len(y_coords)} x {len(x_coords)}")

fig_size = (min(len(x_coords) * 0.2, 12), min(len(y_coords) * 0.2, 10))
plt.figure(figsize=fig_size)
im = plt.imshow(heatmap_count_norm, cmap='hot', aspect='equal', interpolation='nearest', vmin=0, vmax=1)
plt.axis('off')
cbar = plt.colorbar(im, label='Normalized cell count', shrink=0.8)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
plt.title(f'{TARGET_CELL} cell count distribution', fontsize=14)
plt.tight_layout()
plt.savefig(output_heatmap_count, dpi=150, bbox_inches='tight')
print(f"细胞个数热图已保存至: {output_heatmap_count}")
plt.close()  # 关闭当前图形，释放内存


# # ========== 绘制热图2: 基因平均表达量 ==========
# print(f"\n开始分析基因平均表达量，box尺寸: {BOX_SIZE_CD74}x{BOX_SIZE_CD74}")
# heatmap_cd74_avg, x_coords, y_coords = build_heatmap_data(BOX_SIZE_CD74, 'avg_expression')
# cd74_avg_valid = heatmap_cd74_avg[~np.isnan(heatmap_cd74_avg)]
# cd74_avg_min, cd74_avg_max = cd74_avg_valid.min(), cd74_avg_valid.max()
# heatmap_cd74_avg_norm = (heatmap_cd74_avg - cd74_avg_min) / (cd74_avg_max - cd74_avg_min) if cd74_avg_max > cd74_avg_min else np.zeros_like(heatmap_cd74_avg)
# print(f"基因平均表达量 - 原始范围: [{cd74_avg_min:.4f}, {cd74_avg_max:.4f}] -> 归一化到 [0, 1]")
# print(f"网格尺寸: {len(y_coords)} x {len(x_coords)}")
#
# fig_size = (min(len(x_coords) * 0.2, 12), min(len(y_coords) * 0.2, 10))
# plt.figure(figsize=fig_size)
# im = plt.imshow(heatmap_cd74_avg_norm, cmap='hot', aspect='equal', interpolation='nearest', vmin=0, vmax=1)
# plt.axis('off')
# cbar = plt.colorbar(im, label='Normalized avg gene expression', shrink=0.8)
# cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
# plt.title(f'{TARGET_GENE} average expression distribution', fontsize=14)
# plt.tight_layout()
# plt.savefig(output_heatmap_cd74_avg, dpi=150, bbox_inches='tight')
# print(f"基因平均表达量热图已保存至: {output_heatmap_cd74_avg}")
# plt.close()


# ========== 绘制热图3: 基因总表达量 ==========
print(f"\n开始分析基因总表达量，box尺寸: {BOX_SIZE_CD74}x{BOX_SIZE_CD74}")
heatmap_cd74_total, x_coords, y_coords = build_heatmap_data(BOX_SIZE_CD74, 'total_expression')
cd74_total_valid = heatmap_cd74_total[~np.isnan(heatmap_cd74_total)]
cd74_total_min, cd74_total_max = cd74_total_valid.min(), cd74_total_valid.max()
heatmap_cd74_total_norm = (heatmap_cd74_total - cd74_total_min) / (cd74_total_max - cd74_total_min) if cd74_total_max > cd74_total_min else np.zeros_like(heatmap_cd74_total)
print(f"基因总表达量 - 原始范围: [{cd74_total_min:.4f}, {cd74_total_max:.4f}] -> 归一化到 [0, 1]")
print(f"网格尺寸: {len(y_coords)} x {len(x_coords)}")

fig_size = (min(len(x_coords) * 0.2, 12), min(len(y_coords) * 0.2, 10))
plt.figure(figsize=fig_size)
im = plt.imshow(heatmap_cd74_total_norm, cmap='hot', aspect='equal', interpolation='nearest', vmin=0, vmax=1)
plt.axis('off')
cbar = plt.colorbar(im, label='Normalized total gene expression', shrink=0.8)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
plt.title(f'{TARGET_GENE} total expression distribution (sum per box)', fontsize=14)
plt.tight_layout()
plt.savefig(output_heatmap_cd74_total, dpi=150, bbox_inches='tight')
print(f"基因总表达量热图已保存至: {output_heatmap_cd74_total}")
plt.close()
