import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ==================== 子函数：绘制多尺度一致性热图 ====================
def plot_consistency_heatmap(std_col, cell_col, gene_col, boxes, img_path, gene_name, box_size=200, beta=5.0):
    """
    绘制多尺度一致性热图
    基于三个指标的秩排序计算一致性指数
    """

    # 计算一致性指数（使用秩变换，天然消除量纲影响）
    def rank_consistency_index(v1, v2, v3, beta=5.0):
        N = len(v1)
        r1 = pd.Series(v1).rank(method='average').values
        r2 = pd.Series(v2).rank(method='average').values
        r3 = pd.Series(v3).rank(method='average').values
        D = (np.abs(r1 - r2) + np.abs(r1 - r3) + np.abs(r2 - r3)) / 3.0
        return np.exp(-beta * (D / N) ** 2)

    C_values = rank_consistency_index(std_col, cell_col, gene_col, beta)

    # 读取图像
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 绘制热图
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img_rgb)

    cmap = plt.cm.RdBu_r
    for idx, (x, y) in enumerate(boxes):
        rect = Rectangle((x, y), box_size, box_size,
                         facecolor=cmap(C_values[idx]),
                         edgecolor='none', alpha=0.7)
        ax.add_patch(rect)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Consistency index', fontsize=12)

    ax.set_title(f'Multi-scale Consistency Heatmap (T2 value——All cells——Gene {gene_name})', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print(f"总box数量: {len(C_values)}")
    print(f"一致性指数范围: [{C_values.min():.3f}, {C_values.max():.3f}]")
    print(f"高一致box (C>0.8): {np.sum(C_values > 0.8)} 个")
    print(f"低一致box (C<0.2): {np.sum(C_values < 0.2)} 个")
    return C_values


# ==================== 子函数：三高/三低box检测和显示 ====================
def plot_high_low_boxes(std_col, cell_col, gene_col, boxes, img_path, gene_name, box_size=200):
    """
    检测并绘制三高和三低box
    每个尺度内部独立进行三分位划分，然后取交集
    """

    # 直接对原始数据进行三分位划分（不依赖量纲）
    def get_tertile_labels(values):
        """
        将数据分为三等份
        返回每个元素的标签：'high', 'mid', 'low'
        """
        high_thresh = np.percentile(values, 66.7)
        low_thresh = np.percentile(values, 33.3)
        return ['high' if v >= high_thresh
                else 'low' if v <= low_thresh
        else 'mid' for v in values]

    # 三个尺度独立划分
    mri_labels = get_tertile_labels(std_col)
    cell_labels = get_tertile_labels(cell_col)
    gene_labels = get_tertile_labels(gene_col)

    # 寻找三高和三低（取交集）
    high_indices = [i for i in range(len(std_col))
                    if mri_labels[i] == 'high' and
                    cell_labels[i] == 'high' and
                    gene_labels[i] == 'high']

    low_indices = [i for i in range(len(std_col))
                   if mri_labels[i] == 'low' and
                   cell_labels[i] == 'low' and
                   gene_labels[i] == 'low']

    # 读取图像并绘制
    img = cv2.imread(img_path)

    # 标注三高box（红色）
    for idx in high_indices:
        x, y = boxes[idx]
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + box_size, y + box_size), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.rectangle(img, (x, y), (x + box_size, y + box_size), (0, 0, 255), 4)
        cv2.putText(img, 'High', (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 标注三低box（蓝色）
    for idx in low_indices:
        x, y = boxes[idx]
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + box_size, y + box_size), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.rectangle(img, (x, y), (x + box_size, y + box_size), (255, 0, 0), 4)
        cv2.putText(img, 'Low', (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 显示图像
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Triple High/Low Boxes (T2 value——All cells——Gene {gene_name})', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print(f"总box数量: {len(std_col)}")
    print(f"三高box数量: {len(high_indices)}")
    print(f"三低box数量: {len(low_indices)}")
    return high_indices, low_indices


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 参数设置
    file_path = r'D:\PythonCode\mouse_process\cross_scale_mapping_brain\brain_data/Box_features_m3.xlsx'
    img_path = r'H:\Cross-scale_mouse_data\ST_data\tumor_3_brain/Brain_3_aligned_HE.png'
    gene_name = 'Camk2a'  # 请根据需要修改基因名称
    box_size = 200

    # 读取数据
    print("正在加载数据...")
    data = pd.read_excel(file_path)

    # 读取坐标
    x_coords, y_coords = data['x'].values, data['y'].values
    boxes = list(zip(x_coords, y_coords))

    # 读取三个尺度数据
    std_col = data['T2 value'].values  # MRI数据
    cell_col = data['All cells'].values  # 细胞数据（固定列名）
    gene_col = data[gene_name].values  # 基因表达数据
    print(f"数据加载完成，共 {len(boxes)} 个box")

    # 子函数1：绘制一致性热图
    C_values = plot_consistency_heatmap(
        std_col, cell_col, gene_col, boxes, img_path,
        gene_name, box_size
    )

    # 子函数2：绘制三高/三低box
    high_indices, low_indices = plot_high_low_boxes(
        std_col, cell_col, gene_col, boxes, img_path,
        gene_name, box_size
    )
