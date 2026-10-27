import pandas as pd
import cv2
import numpy as np
from PIL import Image
from openpyxl import load_workbook

# ========== 路径配置 ==========
box_path = r"D:\PythonCode\mouse_process\liver_niche\box_locations.xlsx"

he_img_path = "H:/Cross-scale_mouse_data/ST_data/tumor_3_liver/Liver_3_aligned_HE.png"
gene_exp_path = r"H:\Cross-scale_mouse_data\ST_data\tumor_3_liver\tumor_3_liver_target_gene_expression.csv"
cell_locations_path = r"H:\Cross-scale_mouse_data\ST_data\tumor_3_liver\filtered_feature_bc_matrix\cell_locations.tsv"
output_path = r"H:\Cross-scale_mouse_data\ST_data\tumor_3_liver\tumor_3_liver_gene_index.xlsx"

# ========== 读取数据 ==========
df_boxes = pd.read_excel(box_path, sheet_name='mouse3')  # 设置小鼠id
df_gene = pd.read_csv(gene_exp_path)
df_cells = pd.read_csv(cell_locations_path, sep='\t')

# ========== 坐标转换参数（基于病理图像尺寸）==========
img = np.array(Image.open(he_img_path))
h, w = img.shape[:2]
scale_ratio_x, scale_ratio_y = 56500 / w, 20434 / h
df_cells['X_img'] = df_cells['X'] / scale_ratio_x
df_cells['Y_img'] = h - (df_cells['Y'] / scale_ratio_y)

# ========== 定义box信息 ==========
boxes = []  # [(type, x, y), ...]
for _, row in df_boxes.iterrows():
    if row['type'] in ['niche', 'tumor', 'normal']:
        boxes.append((row['type'], row['x'], row['y'], row['x'] + 400, row['y'] + 400))

# ========== 计算每个box的平均表达量 ==========
avg_expressions = []  # 存储每个box的平均表达量

for box_type, x1, y1, x2, y2 in boxes:
    # 找出框内的细胞
    in_box = df_cells[(df_cells['X_img'] >= x1) & (df_cells['X_img'] <= x2) &
                      (df_cells['Y_img'] >= y1) & (df_cells['Y_img'] <= y2)]

    if len(in_box) == 0:
        print(f"空box: {box_type}, 坐标: ({x1},{y1})-({x2},{y2})")
        continue

    barcodes = in_box['Cell_Barcode'].tolist()  # 获取这些细胞的barcode
    cell_exp = df_gene[df_gene['barcode'].isin(barcodes)].copy()  # 从基因表达表中提取对应细胞的表达量
    if len(cell_exp) == 0:
        continue

    # 计算平均表达量（每个基因）
    avg_exp = cell_exp.drop(columns=['barcode']).mean().to_frame().T
    avg_exp.insert(0, 'type', box_type)
    avg_expressions.append(avg_exp)

# # ========== 保存基因表达指标到工作表 ==========
# if avg_expressions:
#     df_avg = pd.concat(avg_expressions, ignore_index=True)
#
#     # 使用 mode='a' 追加模式，不覆盖原有工作表
#     with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#         df_avg.to_excel(writer, sheet_name='average_cell_expression', index=False)
#     print(f"已保存至: {output_path} (新增工作表 'average_cell_expression')")
#     print(f"方框数: {len(avg_expressions)}")
# else:
#     print("没有有效的box数据可保存")


# ========== 统计每个box中特定细胞类型的个数，保存到工作表 ==========
niche_spot_path = r"H:\Cross-scale_mouse_data\ST_data\Niche_spot_coordinates.csv"
df_niche = pd.read_csv(niche_spot_path)
df_niche = df_niche[df_niche['sample'] == 'Liver_3'].copy()  # 只保留sample列为特定小鼠的行

# 提取需要的列：sample、barcode、x、y、dominant_celltype
df_niche = df_niche[['sample', 'barcode', 'x', 'y', 'dominant_celltype']].copy()

# 对niche spot坐标进行相同的坐标转换
df_niche['X_img'] = df_niche['x'] / scale_ratio_x
df_niche['Y_img'] = h - (df_niche['y'] / scale_ratio_y)

# 定义需要统计的五种细胞类型
cell_types_to_count = ['cDC2', 'NK_cells', 'Endothelial', 'HSC_Activated_ECM', 'Macrophage_M1']

# 统计每个box中各种细胞类型的个数
cell_count_data = []
for box_type, x1, y1, x2, y2 in boxes:
    # 找出框内的 spots
    in_box = df_niche[(df_niche['X_img'] >= x1) & (df_niche['X_img'] <= x2) &
                      (df_niche['Y_img'] >= y1) & (df_niche['Y_img'] <= y2)]

    # 统计每种细胞类型的个数
    cell_counts = {'type': box_type, 'location': f"{x1}x{y1}"}
    for cell_type in cell_types_to_count:
        cell_counts[cell_type] = len(in_box[in_box['dominant_celltype'] == cell_type])

    cell_count_data.append(cell_counts)

# 创建DataFrame
df_cell_counts = pd.DataFrame(cell_count_data)

# 保存到新的工作表（追加到已有的Excel文件）
with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_cell_counts.to_excel(writer, sheet_name='cell_count', index=False)

print(f"已保存细胞计数至: {output_path} (新增工作表 'cell_count')")
print(f"统计的方框数: {len(cell_count_data)}")
print(f"统计的细胞类型: {', '.join(cell_types_to_count)}")
