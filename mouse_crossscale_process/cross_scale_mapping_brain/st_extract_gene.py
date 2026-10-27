##############################################
# 读取原始ST数据，先进行标准化，然后提取目标基因表达量保存为csv表格
#（注意：小鼠脑直接提取基因，小鼠肝有两套基因名）
##############################################

import os
import csv
import scipy.io
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

# ==================== 配置路径 ====================
dir_path = 'H:/Cross-scale_mouse_data/ST_data'
eachpath = 'tumor_8_brain'  # 设置哪只小鼠
data_dir = os.path.join(dir_path, eachpath)
matrix_dir = os.path.join(data_dir, "filtered_feature_bc_matrix")

# 输入文件
matrix_file = os.path.join(matrix_dir, "matrix.mtx")
features_file = os.path.join(matrix_dir, "features.tsv")
barcodes_file = os.path.join(matrix_dir, "barcodes.tsv")
spot_list_file = os.path.join(matrix_dir, "cell_locations.tsv")

# 小鼠脑目标基因列表
target_genes = ['Emx1', 'Slc17a7', 'Camk2a', 'Neurod6', 'Nrgn', 'Satb2', 'Cux1', 'Cux2', 'Rorb', 'Bcl11b', 'Tbr1',
                'Fezf2', 'Dlg4', 'Homer1', 'Map2', 'Grin2b', 'Gria1', 'Bdnf', 'Arc', 'Reln', 'Nrn1', 'Lmo4', 'Ntsr1',
                'Adcyap1', 'Gad1', 'Gad2', 'Calb2', 'Vip', 'Htr3a', 'Cldn5', 'Tjp1', 'Col4a1', 'Col4a2', 'Acta2', 'Lpl',
                'Cx3cl1', 'Mapt', 'Fgf12', 'Slc1a2', 'Vsnl1']

# ==================== 1. 读取数据（保持稀疏矩阵） ====================
print("读取基因表达矩阵（稀疏格式）...")
mat = scipy.io.mmread(matrix_file).tocsr().T  # 改为CSR格式，支持索引

# 读取基因名称
gene_names = []
with open(features_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        gene_names.append(row[1])  # 基因名称

# 读取barcodes（spot标识）
barcodes = []
with open(barcodes_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        barcodes.append(row[0])

# 读取组织内spot坐标
spot_coord = pd.read_csv(spot_list_file, sep='\t')
tissue_barcodes = set(spot_coord['Cell_Barcode'].tolist())  # 使用set提高查找效率

# ==================== 2. 筛选组织内的spots ====================
print(f"总spots数: {len(barcodes)}")
print(f"组织内spots数: {len(tissue_barcodes)}")

# 获取组织内spots的索引
tissue_indices = [i for i, bc in enumerate(barcodes) if bc in tissue_barcodes]
print(f"筛选后spots数: {len(tissue_indices)}")
# 筛选稀疏矩阵（行筛选）
if issparse(mat):
    mat_filtered = mat[tissue_indices, :]
else:
    mat_filtered = mat[tissue_indices, :]

# ==================== 3. 找到目标基因的列索引 ====================
gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
target_indices = []
target_gene_names = []
for gene in target_genes:
    if gene in gene_to_idx:
        target_indices.append(gene_to_idx[gene])
        target_gene_names.append(gene)
    else:
        print(f"警告: 基因 {gene} 未找到")

if not target_indices:
    raise ValueError("未找到任何目标基因！")

print(f"找到目标基因: {target_gene_names}")

# 提取目标基因的稀疏矩阵（只保留需要的列）
mat_target = mat_filtered[:, target_indices]

# ==================== 4. 标准化处理（内存高效） ====================
print("进行数据标准化...")

# 转换为CSR格式以便高效计算
if not issparse(mat_target):
    mat_target = csr_matrix(mat_target)
else:
    mat_target = mat_target.tocsr()

# 计算每个细胞的总UMI数（使用稀疏矩阵的行和）
total_umi_per_cell = np.array(mat_target.sum(axis=1)).flatten()

# 避免除以零
total_umi_per_cell = np.maximum(total_umi_per_cell, 1)

# 缩放因子（缩放到1e4）
scale_factor = 10000.0 / total_umi_per_cell

# 逐行进行标准化（避免创建密集矩阵）
# 使用diag矩阵乘法进行缩放
scale_diag = csr_matrix((scale_factor, (range(len(scale_factor)), range(len(scale_factor)))),
                        shape=(len(scale_factor), len(scale_factor)))
mat_normalized = scale_diag @ mat_target

# 应用log1p变换（对非零元素）
mat_normalized.data = np.log1p(mat_normalized.data)

print(f"标准化完成")

# ==================== 5. 提取组织内spots的barcodes ====================
tissue_barcodes_filtered = [barcodes[i] for i in tissue_indices]

# ==================== 6. 转换为DataFrame并保存 ====================
# 将稀疏矩阵转换为密集矩阵
expression_matrix = mat_normalized.toarray()

# 创建DataFrame，将barcode作为第一列
result_df = pd.DataFrame({
    'barcode': tissue_barcodes_filtered,
    **{gene: expression_matrix[:, i] for i, gene in enumerate(target_gene_names)}
})

# 保存为CSV
output_file = os.path.join(data_dir, f"{eachpath}_corss-scale_expression.csv")
result_df.to_csv(output_file, float_format='%.6f', index=False)
print(f"\n目标基因表达矩阵已保存到: {output_file}")
print(f"输出矩阵形状: {result_df.shape}")
