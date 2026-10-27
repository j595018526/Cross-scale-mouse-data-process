"""
病理图像区域采样工具
功能：从肿瘤边缘、肿瘤内部、正常组织三个区域均匀采样400x400的box
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random


# ==================== 核心子函数 ====================

def extract_region_masks(mask_img):
    """根据BGR颜色提取三个区域的二值mask"""
    colors = {
        'tumor_edge': np.array([0, 0, 0]),  # 黑色 - 肿瘤边缘
        'tumor_interior': np.array([36, 28, 237]),  # 偏红色 - 肿瘤内部
        'normal': np.array([0, 242, 255])  # 偏黄色 - 正常组织
    }

    masks = {}
    for region, color in colors.items():
        masks[region] = cv2.inRange(mask_img, color, color)
    return masks


def get_region_bounds(mask):
    """计算mask区域的上下左右边界"""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_min, x_max = np.min(coords[1]), np.max(coords[1])
    return x_min, x_max, y_min, y_max


def get_valid_positions(mask, region_type, box_size=(400, 400), coverage_threshold=0.80):
    """获取区域内所有有效的box位置"""
    h, w = mask.shape
    box_h, box_w = box_size
    valid_positions = []

    for y in range(0, h - box_h + 1, 50):
        for x in range(0, w - box_w + 1, 50):
            box_region = mask[y:y + box_h, x:x + box_w]

            if region_type == 'tumor_edge':
                # 肿瘤边缘：要求95%以上在mask内
                coverage = np.sum(box_region > 0) / (box_h * box_w)
                if coverage >= coverage_threshold:
                    valid_positions.append((x, y))
            else:
                # 肿瘤内部和正常组织：要求99%在mask内
                coverage = np.sum(box_region > 0) / (box_h * box_w)
                if coverage >= 0.99:
                    valid_positions.append((x, y))

    return valid_positions


def uniform_sample_positions(valid_positions, k, mask, box_size=(400, 400), grid_factor=1.8):
    """在mask边界内网格化采样k个box位置

    Args:
        valid_positions: 有效位置列表
        k: 需要采样的数量
        mask: 区域mask
        box_size: box大小
        grid_factor: 网格密度因子（>1表示更密集的网格，推荐1.5-2.0）
    """
    if len(valid_positions) <= k:
        return valid_positions

    # 计算mask区域边界
    bounds = get_region_bounds(mask)
    if bounds is None:
        return []

    x_min, x_max, y_min, y_max = bounds
    box_h, box_w = box_size

    # 调整边界确保box完全在图像内
    x_min = max(x_min, 0)
    x_max = min(x_max, mask.shape[1] - box_w)
    y_min = max(y_min, 0)
    y_max = min(y_max, mask.shape[0] - box_h)

    # 在边界内创建网格（使用grid_factor增加网格密度）
    grid_size = int(np.sqrt(k * grid_factor)) + 1
    grid_x = np.linspace(x_min, x_max, grid_size)
    grid_y = np.linspace(y_min, y_max, grid_size)

    selected = []
    for gy in grid_y:
        for gx in grid_x:
            if len(selected) >= k:
                break

            # 找到距离网格点最近的候选位置
            distances = [np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
                        for x, y in valid_positions]
            if distances:
                nearest_idx = np.argmin(distances)
                nearest_pos = valid_positions[nearest_idx]

                # 验证该box是否在mask区域内
                if nearest_pos not in selected:
                    x, y = nearest_pos
                    box_region = mask[y:y+box_h, x:x+box_w]
                    if np.all(box_region > 0):  # 确保完全在区域内
                        selected.append(nearest_pos)

        if len(selected) >= k:
            break

    # 随机补充不足的部分
    if len(selected) < k:
        remaining = [pos for pos in valid_positions if pos not in selected]
        if remaining:
            additional = random.sample(remaining, min(k - len(selected), len(remaining)))
            selected.extend(additional)

    return selected[:k]


def visualize_boxes(he_img, boxes_dict, save_path):
    """可视化并保存box标注结果"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB))
    color_map = {'tumor_edge': 'red', 'tumor_interior': 'yellow', 'normal': 'green'}
    for region, positions in boxes_dict.items():
        for x, y in positions:
            rect = Rectangle((x, y), 400, 400, linewidth=2, edgecolor=color_map[region], facecolor='none', alpha=0.8)
            ax.add_patch(rect)

    ax.set_title('Extracted Boxes (Red: Tumor Edge, Yellow: Tumor Interior, Green: Normal)')
    ax.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 1. 加载图像
    basepath = r'H:\Cross-scale_mouse_data\ST_data'
    tumor_path = Path(basepath) / 'tumor_3_liver'
    he_img = cv2.imread(str(tumor_path / 'Liver_3_aligned_HE.png'))
    mask_img = cv2.imread(str(tumor_path / 'Liver_3_HE_mask.png'))

    k = 40  # 每个区域抽取的box数量
    grid_factor = 1.2  # 网格密度因子（1.5-2.0之间调整）
    box_size = (400, 400)

    # 创建输出目录
    output_dir = Path('D:/PythonCode')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 提取三个区域的mask
    print("提取区域mask...")
    masks = extract_region_masks(mask_img)

    # 3. 为每个区域提取box
    boxes_dict = {}
    for region, mask in masks.items():
        print(f"\n处理 {region} 区域...")

        # 获取该区域边界
        bounds = get_region_bounds(mask)
        if bounds:
            print(f"区域边界: x=[{bounds[0]}, {bounds[1]}], y=[{bounds[2]}, {bounds[3]}]")

        valid_positions = get_valid_positions(mask, region, box_size)
        print(f"找到 {len(valid_positions)} 个有效位置")

        # 使用改进的采样函数（传入mask用于验证）
        selected_positions = uniform_sample_positions(valid_positions, k, mask, box_size, grid_factor)
        boxes_dict[region] = selected_positions
        print(f"选择了 {len(selected_positions)} 个box")

    # 4. 可视化结果
    print("\n可视化结果...")
    save_path = output_dir / 'box_show_mouse3.png'
    visualize_boxes(he_img, boxes_dict, save_path)

    # 5. 保存坐标到Excel
    data = []
    for region, positions in boxes_dict.items():
        for x, y in positions:
            data.append({'type': region, 'x': x, 'y': y})

    df = pd.DataFrame(data)
    excel_path = output_dir / 'box_position_mouse3.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"结果已保存到: {excel_path}")

    # 6. 统计信息
    print("\n=== 统计信息 ===")
    for region, positions in boxes_dict.items():
        print(f"{region}: {len(positions)}个box")
    print(f"\n可视化图像: {save_path}")
    print(f"坐标数据: {excel_path}")
