import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ========== 配置路径 ==========
base_path = "H:/Cross-scale_mouse_data/ST_data/tumor_8_brain"
he_img_path = f"{base_path}/Brain_8_aligned_HE.png"
mask_path = f"{base_path}/Brain_8_aligned_HE_mask.png"
cell_locations_path = f"{base_path}/filtered_feature_bc_matrix/cell_locations.tsv"
SAMPLE_NAME = 'tumor_8_brain'
BOX_SIZE = 100

# ========== 输出路径 ==========
output_dir = "D:/"
output_path = f"{output_dir}/{SAMPLE_NAME}_all_cells_heatmap_overlay.png"

# ========== 加载数据 ==========
# 1. 读取HE图像
he_img = np.array(Image.open(he_img_path))
he_h, he_w = he_img.shape[:2]

# 2. 读取mask
mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
tissue_mask = cv2.bitwise_not(mask_img)  # 组织区域为255

# 3. 读取细胞位置
df_cells = pd.read_csv(cell_locations_path, sep='\t')

# 4. 坐标转换
scale_ratio_x = 56500 / he_w
scale_ratio_y = 20434 / he_h
df_cells['X_img'] = df_cells['X'] / scale_ratio_x
df_cells['Y_img'] = he_h - (df_cells['Y'] / scale_ratio_y)

# 5. 筛选有效范围内的细胞
df_cells = df_cells[(df_cells['X_img'] >= 0) & (df_cells['X_img'] < he_w) &
                    (df_cells['Y_img'] >= 0) & (df_cells['Y_img'] < he_h)]

print(f"有效细胞数量: {len(df_cells)}")

# ========== 构建热图数据（滑动窗口） ==========
# 计算有效的窗口起始位置（只保留完全在组织内的窗口）
struct_elem = np.ones((BOX_SIZE, BOX_SIZE), dtype=np.uint8)
valid_start_mask = cv2.erode(tissue_mask, struct_elem)
valid_starts = np.column_stack(np.where(valid_start_mask == 255))

# 步长等于box大小，非重叠滑动窗口
step = BOX_SIZE
valid_start_positions = [(x, y) for y, x in valid_starts if y % step == 0 and x % step == 0]
valid_start_positions = sorted(set(valid_start_positions), key=lambda p: (p[1], p[0]))

# 创建与HE图像相同尺寸的热图（初始为0）
heatmap = np.zeros((he_h, he_w), dtype=np.float32)

# 统计每个窗口内的细胞数量并填充到对应位置
for x, y in valid_start_positions:
    # 统计该窗口内的细胞数
    count = len(df_cells[(df_cells['X_img'] >= x) & (df_cells['X_img'] < x + BOX_SIZE) &
                         (df_cells['Y_img'] >= y) & (df_cells['Y_img'] < y + BOX_SIZE)])
    # 在窗口区域内填充该数值
    heatmap[y:y+BOX_SIZE, x:x+BOX_SIZE] = count

# 只保留组织区域的热图值（非组织区域设为0）
heatmap[tissue_mask == 0] = 0

# ========== 归一化热图到0-1范围 ==========
heatmap_valid = heatmap[tissue_mask == 255]  # 只取组织区域的值
if len(heatmap_valid) > 0 and heatmap_valid.max() > 0:
    heatmap_norm = heatmap / heatmap_valid.max()
else:
    heatmap_norm = np.zeros_like(heatmap)

print(f"细胞数量范围: [{heatmap_valid.min():.0f}, {heatmap_valid.max():.0f}]")

# ========== 创建半透明热图叠加 ==========
# 使用turbo colormap生成彩色热图
cmap = plt.cm.turbo
heatmap_colored = cmap(heatmap_norm)[:, :, :3]  # 获取RGB通道

# 将热图转换为uint8格式
heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

# 创建透明度掩码（组织区域透明度0.5，非组织区域完全透明）
alpha = np.zeros((he_h, he_w, 1), dtype=np.uint8)
alpha[tissue_mask == 255] = 128  # 50%透明度

# 合成热图图像（RGBA）
heatmap_rgba = np.concatenate([heatmap_colored, alpha], axis=2)

# 将HE图像转为RGBA
if he_img.shape[2] == 3:
    he_rgba = np.concatenate([he_img, np.full((he_h, he_w, 1), 255, dtype=np.uint8)], axis=2)
else:
    he_rgba = he_img.copy()

# 叠加：使用PIL的composite或直接alpha混合
from PIL import Image

# 转换为PIL图像
he_pil = Image.fromarray(he_rgba, 'RGBA')
heatmap_pil = Image.fromarray(heatmap_rgba, 'RGBA')

# 合成（使用PIL的alpha复合）
overlay = Image.alpha_composite(he_pil, heatmap_pil)

# ========== 保存结果 ==========
overlay.save(output_path)
print(f"叠加热图已保存至: {output_path} (尺寸: {overlay.size})")
