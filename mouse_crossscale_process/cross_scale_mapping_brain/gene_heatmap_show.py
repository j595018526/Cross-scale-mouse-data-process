# """
# 统计box内所有细胞的所有基因（只要在表格中有列数据的）表达总量，绘制基因表达热图
# """
# import cv2, os
# import numpy as np
# import pandas as pd
# from PIL import Image
# import matplotlib.pyplot as plt
# from scipy import ndimage
#
# # ========== 设置中文字体 ==========
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# def build_heatmap_data(box_size, target_gene):
#     """用滑动窗口构建基因总表达量热图数据"""
#     struct_elem = np.ones((box_size, box_size), dtype=np.uint8)
#     valid_start_mask = cv2.erode(tissue_mask, struct_elem)
#     valid_starts = np.column_stack(np.where(valid_start_mask == 255))
#     step = box_size
#     valid_start_positions = [(x, y) for y, x in valid_starts if y % step == 0 and x % step == 0]
#     valid_start_positions = sorted(set(valid_start_positions), key=lambda p: (p[1], p[0]))
#
#     boxes_data = []
#     for x, y in valid_start_positions:
#         in_box = df_cells[(df_cells['X_img'] >= x) & (df_cells['X_img'] < x + box_size) &
#                           (df_cells['Y_img'] >= y) & (df_cells['Y_img'] < y + box_size)]
#         # 计算基因总表达量
#         value = in_box[target_gene].sum() if len(in_box) > 0 else 0
#         boxes_data.append((x, y, value))
#
#     unique_x = sorted(set(x for x, _, _ in boxes_data))
#     unique_y = sorted(set(y for _, y, _ in boxes_data))
#     x_to_idx = {x: i for i, x in enumerate(unique_x)}
#     y_to_idx = {y: i for i, y in enumerate(unique_y)}
#     heatmap = np.full((len(unique_y), len(unique_x)), np.nan)
#     for x, y, val in boxes_data:
#         heatmap[y_to_idx[y], x_to_idx[x]] = val
#     return heatmap, unique_x, unique_y
#
#
# # ========== 载入路径配置 ==========
# base_path = "H:/Cross-scale_mouse_data/ST_data/tumor_3_brain"
# he_img_path = f"{base_path}/Brain_3_aligned_HE.png"
# mask_path = f"{base_path}/Brain_3_aligned_HE_mask.png"
# gene_exp_path = f"{base_path}/tumor_3_brain_corss-scale_expression.csv"
# cell_locations_path = f"{base_path}/filtered_feature_bc_matrix/cell_locations.tsv"
#
# # ========== 参数设置 ==========
# SAMPLE_NAME = 'Brain_3'
# BOX_SIZE = 100
#
# # ========== 输出路径配置 ==========
# output_dir = "D:\PythonCode\mouse_process\image_brain\热图结果\基因热图"
# os.makedirs(output_dir, exist_ok=True)
#
# # ========== 加载数据 ==========
# print("加载数据...")
# mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# tissue_mask = cv2.bitwise_not(mask_img)
# h, w = tissue_mask.shape
# print(f"图像尺寸: {w} x {h}")
#
# # 读取基因表达数据
# df_gene = pd.read_csv(gene_exp_path)
# print(f"基因表达数据: {df_gene.shape}, 基因数: {df_gene.shape[1] - 1}")
#
# # 读取细胞位置数据
# df_cells = pd.read_csv(cell_locations_path, sep='\t')
# he_img = np.array(Image.open(he_img_path))
# he_h, he_w = he_img.shape[:2]
# scale_ratio_x, scale_ratio_y = 56500 / he_w, 20434 / he_h
#
# # 转换坐标
# df_cells['X_img'] = df_cells['X'] / scale_ratio_x
# df_cells['Y_img'] = he_h - (df_cells['Y'] / scale_ratio_y)
# df_cells = df_cells[(df_cells['X_img'] >= 0) & (df_cells['X_img'] < he_w) &
#                     (df_cells['Y_img'] >= 0) & (df_cells['Y_img'] < he_h)]
#
# print(f"数据准备完成，共 {len(df_cells)} 个有效细胞")
#
# # ========== 获取基因列表（第二列到最后一列） ==========
# gene_list = df_gene.columns[1:].tolist()  # 假设第一列是barcode
# print(f"共 {len(gene_list)} 个基因需要处理")
#
# # ========== 循环处理每个基因 ==========
# for idx, target_gene in enumerate(gene_list):
#     print(f"\n处理基因 {idx + 1}/{len(gene_list)}: {target_gene}")
#
#     # 将基因表达数据映射到细胞
#     gene_exp_dict = dict(zip(df_gene['barcode'], df_gene[target_gene]))
#     df_cells[target_gene] = df_cells['Cell_Barcode'].map(gene_exp_dict).fillna(0)
#
#     # 构建热图
#     heatmap, x_coords, y_coords = build_heatmap_data(BOX_SIZE, target_gene)
#
#     # 归一化
#     valid_data = heatmap[~np.isnan(heatmap)]
#     if len(valid_data) > 0:
#         data_min, data_max = valid_data.min(), valid_data.max()
#         heatmap_norm = (heatmap - data_min) / (data_max - data_min) if data_max > data_min else np.zeros_like(heatmap)
#         print(f"  表达范围: [{data_min:.4f}, {data_max:.4f}] -> 归一化到 [0, 1]")
#     else:
#         heatmap_norm = np.zeros_like(heatmap)
#         print(f"  警告: 无有效数据")
#
#     # ========== 上下翻转（垂直翻转） ==========
#     heatmap_norm = np.flipud(heatmap_norm)
#
#     # 绘制并保存
#     fig_size = (min(len(x_coords) * 0.2, 12), min(len(y_coords) * 0.2, 10))
#     plt.figure(figsize=fig_size)
#     im = plt.imshow(heatmap_norm, cmap='turbo', aspect='equal', interpolation='nearest', vmin=0, vmax=1)
#     plt.axis('off')
#     cbar = plt.colorbar(im, label='Normalized total gene expression', shrink=0.8)
#     cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
#     plt.title(f'{target_gene} total expression distribution', fontsize=10)
#     plt.tight_layout()
#
#     # 保存
#     output_path = f"{output_dir}/{SAMPLE_NAME}_{target_gene}.png"
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"  已保存: {output_path}")
#
# print(f"\n所有基因热图生成完成！共生成 {len(gene_list)} 张热图")



"""
统计box内所有细胞的某几个指定基因的表达总量，绘制基因表达热图
"""
import cv2, os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
base_path = "H:/Cross-scale_mouse_data/ST_data/blank_1_brain"
output_dir = "D:\PythonCode\mouse_process\image_brain\热图结果\基因热图/blank1"
os.makedirs(output_dir, exist_ok=True)

# 参数
SAMPLE_NAME = 'Blank_1'
BOX_SIZE = 100
FLIP_UD = False  # True: 上下翻转, False: 不翻转
target_genes = ['Camk2a', 'Dlg4', 'Homer1', 'Map2', 'Grin2b', 'Col4a2']

# 加载数据
print("加载数据...")
tissue_mask = cv2.bitwise_not(cv2.imread(f"{base_path}/Bla_brain_1_aligned_HE_mask.png", cv2.IMREAD_GRAYSCALE))
df_gene = pd.read_csv(f"{base_path}/blank_1_brain_corss-scale_expression.csv")
df_cells = pd.read_csv(f"{base_path}/filtered_feature_bc_matrix/cell_locations.tsv", sep='\t')

# 坐标转换
he_img = np.array(Image.open(f"{base_path}/Bla_brain_1_aligned_HE.png"))
he_h, he_w = he_img.shape[:2]
scale_ratio_x, scale_ratio_y = 56500 / he_w, 20434 / he_h
df_cells['X_img'] = df_cells['X'] / scale_ratio_x
df_cells['Y_img'] = he_h - (df_cells['Y'] / scale_ratio_y)
df_cells = df_cells[(df_cells['X_img'] >= 0) & (df_cells['X_img'] < he_w) &
                    (df_cells['Y_img'] >= 0) & (df_cells['Y_img'] < he_h)]
print(f"有效细胞: {len(df_cells)}")

# 检查可用基因
available_genes = [g for g in target_genes if g in df_gene.columns]
if not available_genes:
    raise ValueError("没有可用的基因!")


# 构建热图函数
def build_heatmap(target_gene):
    struct_elem = np.ones((BOX_SIZE, BOX_SIZE), dtype=np.uint8)
    valid_starts = [(x, y) for y, x in np.column_stack(np.where(cv2.erode(tissue_mask, struct_elem) == 255))
                    if y % BOX_SIZE == 0 and x % BOX_SIZE == 0]
    valid_starts = sorted(set(valid_starts), key=lambda p: (p[1], p[0]))

    gene_dict = dict(zip(df_gene['barcode'], df_gene[target_gene]))
    df_cells[target_gene] = df_cells['Cell_Barcode'].map(gene_dict).fillna(0)

    boxes_data = []
    for x, y in valid_starts:
        in_box = df_cells[(df_cells['X_img'] >= x) & (df_cells['X_img'] < x + BOX_SIZE) &
                          (df_cells['Y_img'] >= y) & (df_cells['Y_img'] < y + BOX_SIZE)]
        boxes_data.append((x, y, in_box[target_gene].sum() if len(in_box) > 0 else 0))

    unique_x, unique_y = sorted(set(x for x, _, _ in boxes_data)), sorted(set(y for _, y, _ in boxes_data))
    heatmap = np.full((len(unique_y), len(unique_x)), np.nan)
    for x, y, val in boxes_data:
        heatmap[unique_y.index(y), unique_x.index(x)] = val

    # 归一化
    valid = heatmap[~np.isnan(heatmap)]
    heatmap_norm = np.zeros_like(heatmap) if len(valid) == 0 or valid.max() == valid.min() else (heatmap - valid.min()) / (valid.max() - valid.min())

    # 根据FLIP_UD控制是否翻转
    if FLIP_UD:
        heatmap_norm = np.flipud(heatmap_norm)
    return heatmap_norm, unique_x, unique_y


# 生成热图
for idx, gene in enumerate(available_genes):
    print(f"\n[{idx + 1}/{len(available_genes)}] 绘制 {gene}...")
    heatmap_norm, x_coords, y_coords = build_heatmap(gene)

    # 绘图
    plt.figure(figsize=(min(len(x_coords) * 0.2, 12), min(len(y_coords) * 0.2, 10)))
    plt.imshow(heatmap_norm, cmap='turbo', aspect='equal', vmin=0, vmax=1)
    plt.axis('off')
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    plt.title(f'{gene} expression', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{SAMPLE_NAME}_{gene}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {SAMPLE_NAME}_{gene}.png")

print(f"\n完成！生成 {len(available_genes)} 张热图")
