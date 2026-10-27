"""
在空间梯度分析中，计算滑动box内的细胞总数和基因表达总量
"""
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# ==================== 路径配置 ====================
base_path = "H:/Cross-scale_mouse_data/ST_data/blank_1_brain"
he_img_path = f"{base_path}/Bla_brain_1_aligned_HE.png"
gene_exp_path = f"{base_path}/blank_1_brain_corss-scale_expression.csv"
cell_locations_path = f"{base_path}/filtered_feature_bc_matrix/cell_locations.tsv"
box_location_path = "./brain_data/gradient_box_location_b1.xlsx"

# ==================== 参数设置 ====================
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
print("读取box位置...")
box_data = pd.read_excel(box_location_path, sheet_name=None)
print(f"找到 {len(box_data)} 个sheet: {list(box_data.keys())}")

# ==================== 统计每个box的特征 ====================
results = {}

for sheet_name, df_boxes in box_data.items():
    print(f"\n处理 {sheet_name}...")

    # 存储结果
    result_list = []

    for _, row in df_boxes.iterrows():
        box_id = row['ID']
        x = int(row['x'])
        y = int(row['y'])

        # 统计该box内的细胞
        in_box = df_cells[(df_cells['X_img'] >= x) & (df_cells['X_img'] < x + BOX_SIZE) &
                         (df_cells['Y_img'] >= y) & (df_cells['Y_img'] < y + BOX_SIZE)]

        # 细胞总数
        cell_count = len(in_box)

        # 各基因表达量（总和）
        gene_expressions = {}
        for gene in available_genes:
            gene_expressions[gene] = in_box[gene].sum() if len(in_box) > 0 else 0

        # 构建结果行
        result_row = {'ID': box_id, 'cell': cell_count}
        result_row.update(gene_expressions)
        result_list.append(result_row)

    # 转换为DataFrame并保存到results字典
    results[sheet_name] = pd.DataFrame(result_list)
    print(f"  {sheet_name}: 统计了 {len(results[sheet_name])} 个box")

# ==================== 保存结果 ====================
output_path = "Sliding_features_b1.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for sheet_name, df_result in results.items():
        df_result.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"已保存 {sheet_name}: {len(df_result)} 行数据")

print(f"\n完成! 结果已保存至: {output_path}")
print(f"包含基因: {available_genes}")
