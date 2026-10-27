"""
在3个区域（海马、丘脑、白质）中，计算每个box内的细胞总数和基因表达总量
"""
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# ==================== 路径配置 ====================
base_path = "H:/Cross-scale_mouse_data/ST_data/tumor_8_brain"
he_img_path = f"{base_path}/Brain_8_aligned_HE.png"
gene_exp_path = f"{base_path}/tumor_8_brain_corss-scale_expression.csv"
cell_locations_path = f"{base_path}/filtered_feature_bc_matrix/cell_locations.tsv"
box_location_path = "./brain_data/box_positions.xlsx"

# ==================== 参数设置 ====================
SHEET_NAME = "m8"  # 指定工作表名称---对应小鼠
BOX_SIZE = 200

# 10个目标基因
target_genes = ['Camk2a', 'Dlg4', 'Homer1', 'Map2', 'Grin2b', 'Col4a2', 'Cx3cl1', 'Mapt', 'Fgf12', 'Vsnl1']

# ==================== 加载数据 ====================
print("加载数据...")

# 1. 读取HE图像获取尺寸
he_img = np.array(Image.open(he_img_path))
he_h, he_w = he_img.shape[:2]

# 2. 读取细胞位置
df_cells = pd.read_csv(cell_locations_path, sep='\t')

# 3. 坐标转换
scale_ratio_x = 56500 / he_w
scale_ratio_y = 20434 / he_h
df_cells['X_img'] = df_cells['X'] / scale_ratio_x
df_cells['Y_img'] = he_h - (df_cells['Y'] / scale_ratio_y)

# 4. 筛选有效范围内的细胞
df_cells = df_cells[(df_cells['X_img'] >= 0) & (df_cells['X_img'] < he_w) &
                    (df_cells['Y_img'] >= 0) & (df_cells['Y_img'] < he_h)]

print(f"有效细胞数量: {len(df_cells)}")

# 5. 读取基因表达数据并映射到细胞
df_gene = pd.read_csv(gene_exp_path)
print(f"基因表达数据: {df_gene.shape}")

# 只保留目标基因
available_genes = [g for g in target_genes if g in df_gene.columns]
print(f"可用目标基因: {len(available_genes)}/{len(target_genes)}")

# 将基因表达映射到细胞
for gene in available_genes:
    gene_exp_dict = dict(zip(df_gene['barcode'], df_gene[gene]))
    df_cells[gene] = df_cells['Cell_Barcode'].map(gene_exp_dict).fillna(0)

# ==================== 读取box位置 ====================
print(f"读取box位置 (工作表: {SHEET_NAME})...")
df_boxes = pd.read_excel(box_location_path, sheet_name=SHEET_NAME)
print(f"加载了 {len(df_boxes)} 个box")

# ==================== 统计每个box的特征 ====================
print("统计box特征...")
result_list = []

for _, row in df_boxes.iterrows():
    box_type = row['type']
    x = int(row['x'])
    y = int(row['y'])

    # 统计该box内的细胞
    in_box = df_cells[(df_cells['X_img'] >= x) & (df_cells['X_img'] < x + BOX_SIZE) &
                      (df_cells['Y_img'] >= y) & (df_cells['Y_img'] < y + BOX_SIZE)]

    # 细胞总数
    cell_count = len(in_box)

    # 各基因表达量（总和）
    gene_expressions = {gene: in_box[gene].sum() for gene in available_genes}

    # 构建结果行 - 将 'cell' 改为 'All cells'
    result_row = {'type': box_type, 'x': x, 'y': y, 'All cells': cell_count}
    result_row.update(gene_expressions)
    result_list.append(result_row)

# 转换为DataFrame
df_result = pd.DataFrame(result_list)
print(f"统计了 {len(df_result)} 个box")

# ==================== 保存结果 ====================
output_path = "Box_features_m8.xlsx"
df_result.to_excel(output_path, sheet_name=SHEET_NAME, index=False)
print(f"\n完成! 结果已保存至: {output_path}")
print(f"包含基因: {available_genes}")
print(f"结果列: {list(df_result.columns)}")
