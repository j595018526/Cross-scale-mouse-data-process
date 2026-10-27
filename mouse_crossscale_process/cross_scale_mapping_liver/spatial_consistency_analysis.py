############################################## 绘制多尺度一致性热图 ###############################################
import cv2
import pandas as pd
import numpy as np

# 1. 读取数据并归一化
file_path = r'D:\PythonCode\mouse_process\cross_scale_mapping\tumor_8_liver_features.xlsx'
mri_data = pd.read_excel(file_path, sheet_name='MRI')
cell_data = pd.read_excel(file_path, sheet_name='cell_count')
expression_data = pd.read_excel(file_path, sheet_name='total_expression')

cell_name = 'HSC_Activated_ECM'
gene_name = 'mm10___Runx1'

std_col = mri_data['std'].values
cell_col = cell_data[cell_name].values
gene_col = expression_data[gene_name].values

# 归一化
std_norm = (std_col - std_col.min()) / (std_col.max() - std_col.min())
cell_norm = (cell_col - cell_col.min()) / (cell_col.max() - cell_col.min())
gene_norm = (gene_col - gene_col.min()) / (gene_col.max() - gene_col.min())

# 2. 每个指标内部划分高/中/低三部分（基于百分位数）
def get_tertile_labels(values):
    """返回每个值的标签: 'high', 'mid', 'low'"""
    high_thresh = np.percentile(values, 66.7)  # 前1/3为高
    low_thresh = np.percentile(values, 33.3)   # 后1/3为低
    labels = []
    for v in values:
        if v >= high_thresh:
            labels.append('high')
        elif v <= low_thresh:
            labels.append('low')
        else:
            labels.append('mid')
    return labels

mri_labels = get_tertile_labels(std_norm)
cell_labels = get_tertile_labels(cell_norm)
gene_labels = get_tertile_labels(gene_norm)

# 3. 寻找三个指标共同的三高和三低box
n = len(std_norm)
high_indices = []  # 三个指标都是high的box
low_indices = []   # 三个指标都是low的box

for i in range(n):
    if mri_labels[i] == 'high' and cell_labels[i] == 'high' and gene_labels[i] == 'high':
        high_indices.append(i)
    elif mri_labels[i] == 'low' and cell_labels[i] == 'low' and gene_labels[i] == 'low':
        low_indices.append(i)

# 4. 读取location坐标
locations = cell_data['location'].values
boxes = []
for loc in locations:
    x, y = map(int, loc.split('x'))
    boxes.append((x, y))

# 5. 读取病理图像
img_path = r'H:\Cross-scale_mouse_data\ST_data\tumor_8_liver/Liver_8_aligned_HE.png'
img = cv2.imread(img_path)

# 6. 绘制结果（只显示三高和三低，其他不显示）
box_size = 400
for idx in high_indices:
    x, y = boxes[idx]
    # 红色填充（半透明）+ 红色边框
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+box_size, y+box_size), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.rectangle(img, (x, y), (x+box_size, y+box_size), (0, 0, 255), 4)
    cv2.putText(img, 'High', (x+10, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

for idx in low_indices:
    x, y = boxes[idx]
    # 蓝色填充（半透明）+ 蓝色边框
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+box_size, y+box_size), (255, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.rectangle(img, (x, y), (x+box_size, y+box_size), (255, 0, 0), 4)
    cv2.putText(img, 'Low', (x+10, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# # 7. 添加图例
# legend_x, legend_y = 20, 50
# cv2.rectangle(img, (legend_x, legend_y), (legend_x+40, legend_y+40), (0, 0, 255), -1)
# cv2.putText(img, 'Triple High (MRI, Cell, Gene all high)', (legend_x+50, legend_y+25),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
# cv2.rectangle(img, (legend_x, legend_y+50), (legend_x+40, legend_y+90), (255, 0, 0), -1)
# cv2.putText(img, 'Triple Low (MRI, Cell, Gene all low)', (legend_x+50, legend_y+75),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# 8. 打印统计信息
print(f"总box数量: {n}")
print(f"三高box数量: {len(high_indices)}")
print(f"三低box数量: {len(low_indices)}")
print(f"\n三高box索引: {high_indices}")
print(f"三低box索引: {low_indices}")

# 9. 保存结果
output_path = 'D:/High_low_mapping_HSC_m8.png'
cv2.imwrite(output_path, img)
print(f"\n图像已保存为 {output_path}")




# # ############################################## 三高/三低box检测和显示 ###############################################
# import cv2
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
# from matplotlib.patches import Rectangle
#
# # ==================== 1. 数据读取 ====================
# file_path = r'D:\PythonCode\mouse_process\cross_scale_mapping\tumor_8_liver_features.xlsx'
# mri_data = pd.read_excel(file_path, sheet_name='MRI')
# cell_data = pd.read_excel(file_path, sheet_name='cell_count')
# expression_data = pd.read_excel(file_path, sheet_name='total_expression')
#
# cell_name = 'HSC_Activated_ECM'
# gene_name = 'mm10___Runx1'
#
# std_col = mri_data['std'].values
# cell_col = cell_data[cell_name].values
# gene_col = expression_data[gene_name].values
#
#
# def rank_consistency_index(v1, v2, v3, beta=5.0):
#     """
#     计算三个指标在同一个box上的多尺度一致性指数
#     参数:
#         v1, v2, v3: 三个尺度的原始数值向量（长度均为N）
#         beta: 灵敏度参数（默认5.0，越大对差异越敏感）
#
#     返回:
#         C: 每个box的一致性指数，范围(0, 1]，1表示完全一致
#     """
#     N = len(v1)
#
#     # 1. 各自独立排序，获取秩次（平均秩处理相等值）
#     r1 = pd.Series(v1).rank(method='average').values
#     r2 = pd.Series(v2).rank(method='average').values
#     r3 = pd.Series(v3).rank(method='average').values
#
#     # 2. 计算两两绝对差异的平均值
#     d12 = np.abs(r1 - r2)
#     d13 = np.abs(r1 - r3)
#     d23 = np.abs(r2 - r3)
#     D = (d12 + d13 + d23) / 3.0
#
#     # 3. 指数衰减转换为[0,1]一致性指数
#     C = np.exp(-beta * (D / N) ** 2)
#     return C
#
#
# # 计算一致性指数
# C_values = rank_consistency_index(std_col, cell_col, gene_col, beta=5.0)
#
# # ==================== 3. 读取位置和图像 ====================
# locations = cell_data['location'].values
# boxes = []
# for loc in locations:
#     x, y = map(int, loc.split('x'))
#     boxes.append((x, y))
#
# img_path = r'H:\Cross-scale_mouse_data\ST_data\tumor_8_liver/Liver_8_aligned_HE.png'
# img = cv2.imread(img_path)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB用于matplotlib显示
#
# # ==================== 4. 在病理图像上绘制带颜色的box ====================
# box_size = 400
# fig, ax = plt.subplots(figsize=(12, 10))
# ax.imshow(img_rgb)
#
# cmap = plt.cm.RdBu_r  # colorbar色系
# for idx, (x, y) in enumerate(boxes):
#     c_value = C_values[idx]
#     color = cmap(c_value)  # c_value在0-1之间
#     rect = Rectangle((x, y), box_size, box_size,
#                      facecolor=color,
#                      edgecolor='none',
#                      alpha=0.7)
#     ax.add_patch(rect)
#
# # ==================== 5. 修改colorbar设置 ====================
# # 创建ScalarMappable对象
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
# sm.set_array([])
#
# # 方法1：使用make_axes_locatable让colorbar与图像等高等宽
# from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# # 创建colorbar的轴，使其高度与主图匹配
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="3%", pad=0.05)  # size控制colorbar宽度，pad控制间距
# cbar = plt.colorbar(sm, cax=cax)
# cbar.set_label('Consistency index', fontsize=12)
#
# # 设置标题
# ax.set_title(f'Multi-scale Consistency Heatmap (Cell: {cell_name}, Gene: {gene_name})', fontsize=14)
# ax.axis('off')
# plt.tight_layout()
# # plt.savefig('D:/consistency_heatmap_HSC_Runx1.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # ==================== 6. 统计输出 ====================
# print(f"总box数量: {len(C_values)}")
# print(f"一致性指数范围: [{C_values.min():.3f}, {C_values.max():.3f}]")
# print(f"一致性指数均值: {C_values.mean():.3f} ± {C_values.std():.3f}")
# print(f"\n高一致box (C>0.8): {np.sum(C_values > 0.8)} 个")
# print(f"低一致box (C<0.2): {np.sum(C_values < 0.2)} 个")
#
# # 输出top 5高一致性box
# top5_idx = np.argsort(C_values)[-5:][::-1]
# print(f"\nTop 5 高一致性box索引: {top5_idx}")
# print(f"对应C值: {C_values[top5_idx]}")
#
