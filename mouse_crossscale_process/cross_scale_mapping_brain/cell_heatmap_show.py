# import cv2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
#
#
# def plot_heatmap_for_celltype(cell_type, df_subset):
#     # 统计每个窗口内的细胞数
#     boxes_data = []
#     for x, y in valid_start_positions:
#         count = len(df_subset[(df_subset['X_img'] >= x) & (df_subset['X_img'] < x + BOX_SIZE) &
#                               (df_subset['Y_img'] >= y) & (df_subset['Y_img'] < y + BOX_SIZE)])
#         boxes_data.append((x, y, count))
#
#     # 构建网格矩阵
#     heatmap = np.full((len(unique_y), len(unique_x)), np.nan)
#     for x, y, val in boxes_data:
#         heatmap[y_to_idx[y], x_to_idx[x]] = val
#
#     # 归一化
#     heatmap_valid = heatmap[~np.isnan(heatmap)]
#     if len(heatmap_valid) == 0:
#         print(f"警告: {cell_type} 无有效数据，跳过")
#         return
#     min_val, max_val = heatmap_valid.min(), heatmap_valid.max()
#     heatmap_norm = (heatmap - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(heatmap)
#
#     # 垂直翻转（上下颠倒）
#     heatmap_norm = np.flipud(heatmap_norm)
#     # 绘制
#     fig_size = (min(len(unique_x) * 0.2, 12), min(len(unique_y) * 0.2, 10))
#     plt.figure(figsize=fig_size)
#     im = plt.imshow(heatmap_norm, cmap='turbo', aspect='equal', interpolation='nearest')
#     plt.axis('off')
#     cbar = plt.colorbar(im, label='Normalized cell count', shrink=0.8)
#     cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
#     plt.title(f'{SAMPLE_NAME} - {cell_type}', fontsize=14)
#     plt.tight_layout()
#
#     save_path = f"{output_dir}/{SAMPLE_NAME}-{cell_type}_heatmap.png"
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"已保存: {save_path}, 窗口数: {len(boxes_data)}, 细胞数: {len(df_subset)}")
#
#
# # ========== 新增：绘制所有细胞热图 ==========
# def plot_heatmap_for_all_cells(df_merged):
#     """
#     统计所有细胞的box内数量，绘制所有细胞的热图
#     """
#     print("\n开始绘制所有细胞热图...")
#
#     # 统计每个窗口内的所有细胞数
#     boxes_data = []
#     for x, y in valid_start_positions:
#         count = len(df_merged[(df_merged['X_img'] >= x) & (df_merged['X_img'] < x + BOX_SIZE) &
#                               (df_merged['Y_img'] >= y) & (df_merged['Y_img'] < y + BOX_SIZE)])
#         boxes_data.append((x, y, count))
#
#     # 构建网格矩阵
#     heatmap = np.full((len(unique_y), len(unique_x)), np.nan)
#     for x, y, val in boxes_data:
#         heatmap[y_to_idx[y], x_to_idx[x]] = val
#
#     # 归一化
#     heatmap_valid = heatmap[~np.isnan(heatmap)]
#     if len(heatmap_valid) == 0:
#         print("警告: 无有效数据，跳过")
#         return
#     min_val, max_val = heatmap_valid.min(), heatmap_valid.max()
#     heatmap_norm = (heatmap - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(heatmap)
#
#     # 垂直翻转（上下颠倒）
#     heatmap_norm = np.flipud(heatmap_norm)
#
#     # 绘制
#     fig_size = (min(len(unique_x) * 0.2, 12), min(len(unique_y) * 0.2, 10))
#     plt.figure(figsize=fig_size)
#     im = plt.imshow(heatmap_norm, cmap='turbo', aspect='equal', interpolation='nearest')
#     plt.axis('off')
#     cbar = plt.colorbar(im, label='Normalized cell count', shrink=0.8)
#     cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
#     plt.title(f'{SAMPLE_NAME} - All Cells', fontsize=14)
#     plt.tight_layout()
#
#     save_path = f"{output_dir}/{SAMPLE_NAME}-All_Cells_heatmap.png"
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"已保存: {save_path}, 窗口数: {len(boxes_data)}, 总细胞数: {len(df_merged)}")
#
#
# # ========== 设置中文字体 ==========
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
# # ========== 配置路径 ==========
# base_path = "H:/Cross-scale_mouse_data/ST_data/tumor_3_brain"
# he_img_path = f"{base_path}/Brain_3_aligned_HE.png"
# mask_path = f"{base_path}/Brain_3_aligned_HE_mask.png"
# cell_locations_path = f"{base_path}/filtered_feature_bc_matrix/cell_locations.tsv"
# celltype_path = "H:/Cross-scale_mouse_data/ST_data/all_brain_celltype.tsv"
# SAMPLE_NAME = '25090559_Brain_3_expression'
# BOX_SIZE = 100
#
# # ========== 输出路径 ==========
# output_dir = "D:/PythonCode/mouse_process/image_brain/热图结果/细胞热图"
#
# # ========== 加载数据 ==========
# # 1. 读取mask
# mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# tissue_mask = cv2.bitwise_not(mask_img)
#
# # 2. 读取细胞类型数据
# df_celltype = pd.read_csv(celltype_path, sep='\t')
# df_celltype = df_celltype[df_celltype['orig.ident'] == SAMPLE_NAME]
#
# # 处理barcode：只取_前面的部分
# df_celltype['barcode'] = df_celltype['barcode'].str.split('_').str[0]
#
# # 3. 读取细胞位置数据
# df_locations = pd.read_csv(cell_locations_path, sep='\t')
#
# # 4. 通过barcode合并数据
# df_merged = df_celltype.merge(df_locations, left_on='barcode', right_on='Cell_Barcode')
#
# # 5. 读取HE图像获取尺寸
# he_img = np.array(Image.open(he_img_path))
# he_h, he_w = he_img.shape[:2]
#
# # 6. 坐标转换
# scale_ratio_x = 56500 / he_w
# scale_ratio_y = 20434 / he_h
# df_merged['X_img'] = df_merged['X'] / scale_ratio_x
# df_merged['Y_img'] = he_h - (df_merged['Y'] / scale_ratio_y)
#
# # 7. 筛选有效范围内的点
# df_merged = df_merged[(df_merged['X_img'] >= 0) & (df_merged['X_img'] < he_w) &
#                       (df_merged['Y_img'] >= 0) & (df_merged['Y_img'] < he_h)]
#
# print(f"有效细胞数量: {len(df_merged)}")
#
# # ========== 计算有效窗口位置 ==========
# struct_elem = np.ones((BOX_SIZE, BOX_SIZE), dtype=np.uint8)
# valid_start_mask = cv2.erode(tissue_mask, struct_elem)
# step = BOX_SIZE
# valid_starts = np.column_stack(np.where(valid_start_mask == 255))
# valid_start_positions = [(x, y) for y, x in valid_starts if y % step == 0 and x % step == 0]
# valid_start_positions = sorted(set(valid_start_positions), key=lambda p: (p[1], p[0]))
#
# # 构建网格索引
# unique_x = sorted(set(x for x, _ in valid_start_positions))
# unique_y = sorted(set(y for _, y in valid_start_positions))
# x_to_idx = {x: i for i, x in enumerate(unique_x)}
# y_to_idx = {y: i for i, y in enumerate(unique_y)}
# print(f"网格尺寸: {len(unique_x)} x {len(unique_y)}")
#
# # ========== 获取所有细胞类型并打印统计信息 ==========
# cell_types = df_merged['all_brain_astro_neuron'].unique()
# print(f"\n共检测到 {len(cell_types)} 种细胞类型:")
# for i, cell_type in enumerate(cell_types, 1):
#     count = len(df_merged[df_merged['all_brain_astro_neuron'] == cell_type])
#     print(f"  {i}. {cell_type}: {count} 个细胞")
#
# print("\n开始绘制热图...")
#
# # ========== 对每一类细胞绘制热图 ==========
# for cell_type in cell_types:
#     df_subset = df_merged[df_merged['all_brain_astro_neuron'] == cell_type]
#     plot_heatmap_for_celltype(cell_type, df_subset)
#
# # ========== 新增：绘制所有细胞热图 ==========
# plot_heatmap_for_all_cells(df_merged)
# # ========== 新增结束 ==========
#
# print("\n全部热图生成完成!")




"""
统计box内所有细胞数量，绘制细胞总量热图
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ========== 配置 ==========
base_path = "H:/Cross-scale_mouse_data/ST_data/blank_1_brain"
he_img_path = f"{base_path}/Bla_brain_1_aligned_HE.png"
mask_path = f"{base_path}/Bla_brain_1_aligned_HE_mask.png"
cell_locations_path = f"{base_path}/filtered_feature_bc_matrix/cell_locations.tsv"
SAMPLE_NAME = 'Brain_blank1'
FLIP_UD = False  # 控制垂直翻转
BOX_SIZE = 100
output_path = f"D:/PythonCode/mouse_process/image_brain/热图结果/{SAMPLE_NAME}_all_cells_heatmap.png"

# ========== 加载数据 ==========
mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
tissue_mask = cv2.bitwise_not(mask_img)
df_cells = pd.read_csv(cell_locations_path, sep='\t')
he_img = np.array(Image.open(he_img_path))
he_h, he_w = he_img.shape[:2]

# 坐标转换
scale_ratio_x = 56500 / he_w
scale_ratio_y = 20434 / he_h
df_cells['X_img'] = df_cells['X'] / scale_ratio_x
df_cells['Y_img'] = he_h - (df_cells['Y'] / scale_ratio_y)

# 筛选有效细胞
df_cells = df_cells[(df_cells['X_img'] >= 0) & (df_cells['X_img'] < he_w) &
                    (df_cells['Y_img'] >= 0) & (df_cells['Y_img'] < he_h)]
print(f"有效细胞数量: {len(df_cells)}")

# ========== 构建热图（滑动窗口） ==========
struct_elem = np.ones((BOX_SIZE, BOX_SIZE), dtype=np.uint8)
valid_start_mask = cv2.erode(tissue_mask, struct_elem)
valid_starts = np.column_stack(np.where(valid_start_mask == 255))

step = BOX_SIZE
valid_start_positions = [(x, y) for y, x in valid_starts if y % step == 0 and x % step == 0]
valid_start_positions = sorted(set(valid_start_positions), key=lambda p: (p[1], p[0]))

# 统计窗口细胞数
boxes_data = []
for x, y in valid_start_positions:
    count = len(df_cells[(df_cells['X_img'] >= x) & (df_cells['X_img'] < x + BOX_SIZE) &
                         (df_cells['Y_img'] >= y) & (df_cells['Y_img'] < y + BOX_SIZE)])
    boxes_data.append((x, y, count))

# 构建网格
unique_x = sorted(set(x for x, _, _ in boxes_data))
unique_y = sorted(set(y for _, y, _ in boxes_data))
x_to_idx = {x: i for i, x in enumerate(unique_x)}
y_to_idx = {y: i for i, y in enumerate(unique_y)}

heatmap = np.full((len(unique_y), len(unique_x)), np.nan)
for x, y, val in boxes_data:
    heatmap[y_to_idx[y], x_to_idx[x]] = val

# ========== 归一化并翻转 ==========
heatmap_valid = heatmap[~np.isnan(heatmap)]
min_val, max_val = heatmap_valid.min(), heatmap_valid.max()
heatmap_norm = (heatmap - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(heatmap)
heatmap_norm = np.flipud(heatmap_norm) if FLIP_UD else heatmap_norm  # 条件翻转

# ========== 绘图保存 ==========
plt.figure(figsize=(min(len(unique_x) * 0.2, 12), min(len(unique_y) * 0.2, 10)))
im = plt.imshow(heatmap_norm, cmap='turbo', aspect='equal', interpolation='nearest')
plt.axis('off')
cbar = plt.colorbar(im, label='Normalized cell count', shrink=0.8)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"热图已保存至: {output_path}")

