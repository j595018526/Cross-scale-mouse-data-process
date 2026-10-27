import pandas as pd
import numpy as np
from PIL import Image

# ========== 载入路径配置 ==========
he_img_path = "H:/Cross-scale_mouse_data/ST_data/tumor_3_liver/Liver_3_aligned_HE.png"
gene_exp_path = r"H:\Cross-scale_mouse_data\ST_data\tumor_3_liver\tumor_3_liver_target_gene_expression.csv"
cell_locations_path = r"H:\Cross-scale_mouse_data\ST_data\tumor_3_liver\filtered_feature_bc_matrix\cell_locations.tsv"

# ========== 输出路径配置 ==========
output_path = r"tumor_3_liver_features.xlsx"

# ========== 读取box坐标数据 ==========
box_path = r"D:\PythonCode\mouse_process\cross_scale_mapping\box_locations_new.xlsx"
df_boxes = pd.read_excel(box_path, sheet_name='mouse3')  # 指定小鼠id
df_gene = pd.read_csv(gene_exp_path)
df_cells = pd.read_csv(cell_locations_path, sep='\t')

# ========== 坐标转换 ==========
img = np.array(Image.open(he_img_path))
h, w = img.shape[:2]
scale_ratio_x, scale_ratio_y = 56500 / w, 20434 / h
df_cells['X_img'] = df_cells['X'] / scale_ratio_x
df_cells['Y_img'] = h - (df_cells['Y'] / scale_ratio_y)

# ========== 定义box信息 ==========
boxes = []
for _, row in df_boxes.iterrows():
    if row['type'] in ['tumor_edge', 'tumor_interior', 'normal']:
        boxes.append((row['type'], row['x'], row['y'], row['x'] + 400, row['y'] + 400))

# ========== 计算每个box的基因总表达量 ==========
total_expressions = []
for box_type, x1, y1, x2, y2 in boxes:
    in_box = df_cells[(df_cells['X_img'] >= x1) & (df_cells['X_img'] <= x2) & (df_cells['Y_img'] >= y1) & (df_cells['Y_img'] <= y2)]
    if len(in_box) == 0:
        print(f"空box: {box_type}, 坐标: ({x1},{y1})-({x2},{y2})")
        continue

    barcodes = in_box['Cell_Barcode'].tolist()
    cell_exp = df_gene[df_gene['barcode'].isin(barcodes)].copy()
    if len(cell_exp) == 0:
        continue

    # 计算总表达量（求和）
    total_exp = cell_exp.drop(columns=['barcode']).sum().to_frame().T
    total_exp.insert(0, 'type', box_type)
    total_exp.insert(1, 'location', f"{x1}x{y1}")  # 添加location列，放在第二列
    total_expressions.append(total_exp)

# ========== 保存基因总表达量 ==========
if total_expressions:
    df_total = pd.concat(total_expressions, ignore_index=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_total.to_excel(writer, sheet_name='total_expression', index=False)
    print(f"已保存总表达量至: {output_path} (方框数: {len(total_expressions)})")
else:
    print("没有有效的box数据可保存")

# ========== 统计每个box中特定细胞类型的个数 ==========
niche_spot_path = r"H:\Cross-scale_mouse_data\ST_data\Niche_spot_coordinates.csv"
df_niche = pd.read_csv(niche_spot_path)
df_niche = df_niche[df_niche['sample'] == 'Liver_3'].copy()   # 指定小鼠id
df_niche = df_niche[['sample', 'barcode', 'x', 'y', 'dominant_celltype']].copy()
df_niche['X_img'] = df_niche['x'] / scale_ratio_x
df_niche['Y_img'] = h - (df_niche['y'] / scale_ratio_y)
cell_types_to_count = ['cDC2', 'NK_cells', 'Endothelial', 'HSC_Activated_ECM', 'Macrophage_M1']
cell_count_data = []
for box_type, x1, y1, x2, y2 in boxes:
    in_box = df_niche[(df_niche['X_img'] >= x1) & (df_niche['X_img'] <= x2) & (df_niche['Y_img'] >= y1) & (df_niche['Y_img'] <= y2)]
    cell_counts = {'type': box_type, 'location': f"{x1}x{y1}"}
    for cell_type in cell_types_to_count:
        cell_counts[cell_type] = len(in_box[in_box['dominant_celltype'] == cell_type])
    cell_count_data.append(cell_counts)

df_cell_counts = pd.DataFrame(cell_count_data)
with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_cell_counts.to_excel(writer, sheet_name='cell_count', index=False)
print(f"已保存细胞计数至: {output_path} (新增工作表 'cell_count')")
print(f"统计的方框数: {len(cell_count_data)}")
print(f"统计的细胞类型: {', '.join(cell_types_to_count)}")
