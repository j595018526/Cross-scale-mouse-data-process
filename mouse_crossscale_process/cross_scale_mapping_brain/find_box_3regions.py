"""
病理图像区域box采样工具
功能：从hip、tha、white三个区域均匀采样200x200的box
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
        'hip': np.array([0, 0, 0]),        # 黑色 - hip
        'white': np.array([36, 28, 237]),  # 偏红色 - white
        'tha': np.array([0, 242, 255])     # 偏黄色 - tha
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


def get_valid_positions(mask, region_type, box_size=(200, 200), coverage_threshold=0.65):
    """获取区域内所有有效的box位置"""
    h, w = mask.shape
    box_h, box_w = box_size
    valid_positions = []

    for y in range(0, h - box_h + 1, 50):
        for x in range(0, w - box_w + 1, 50):
            box_region = mask[y:y + box_h, x:x + box_w]

            if region_type == 'hip':
                # hip：要求coverage_threshold以上在mask内
                coverage = np.sum(box_region > 0) / (box_h * box_w)
                if coverage >= coverage_threshold:
                    valid_positions.append((x, y))
            else:
                # tha和white：要求96%在mask内
                coverage = np.sum(box_region > 0) / (box_h * box_w)
                if coverage >= 0.95:
                    valid_positions.append((x, y))

    return valid_positions


def uniform_sample_positions(valid_positions, k, mask, box_size=(200, 200)):
    """
    修改后的采样函数：使用分层网格 + 距离约束，确保分布均匀
    """
    if len(valid_positions) <= k:
        return valid_positions

    # 1. 计算边界
    bounds = get_region_bounds(mask)
    if bounds is None:
        return []
    x_min, x_max, y_min, y_max = bounds
    box_h, box_w = box_size

    # 边界裁剪
    x_min = max(x_min, 0)
    x_max = min(x_max, mask.shape[1] - box_w)
    y_min = max(y_min, 0)
    y_max = min(y_max, mask.shape[0] - box_h)

    # 2. 生成网格点（使用略微稀疏的网格，避免重复选中）
    # 根据k和区域大小动态计算网格数量，保证每个格子至少能分到一个点
    region_w, region_h = x_max - x_min, y_max - y_min
    grid_count = max(3, int(np.sqrt(k))) + 1  # 根据k计算网格线数量
    grid_x = np.linspace(x_min, x_max, grid_count)
    grid_y = np.linspace(y_min, y_max, grid_count)

    selected = []
    used_indices = set() # 记录已选择的索引，防止重复选中同一位置
    candidate_indices = list(range(len(valid_positions)))
    random.shuffle(candidate_indices) # 打乱，防止顺序偏差

    # 3. 核心逻辑：为每个网格选一个“未使用过”的有效点
    for gy in grid_y:
        for gx in grid_x:
            if len(selected) >= k:
                break

            # 寻找离该网格点最近的、且未被选中的有效点
            best_idx = -1
            best_dist = float('inf')

            for idx in candidate_indices:
                if idx in used_indices:
                    continue
                x, y = valid_positions[idx]
                dist = (x - gx)**2 + (y - gy)**2  # 使用平方距离，避免开根号
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            # 如果找到了合适的点
            if best_idx != -1:
                x, y = valid_positions[best_idx]
                selected.append((x, y))
                used_indices.add(best_idx)

        if len(selected) >= k:
            break

    # 4. 如果网格不够，从剩余池子里随机补足
    if len(selected) < k:
        remaining_idx = [i for i in candidate_indices if i not in used_indices]
        if remaining_idx:
            pick = min(k - len(selected), len(remaining_idx))
            add_indices = random.sample(remaining_idx, pick)
            for idx in add_indices:
                selected.append(valid_positions[idx])

    return selected[:k]


def visualize_boxes(he_img, boxes_dict, save_path=False):
    """可视化并保存box标注结果"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB))

    # 颜色映射：与mask标注颜色保持一致
    color_map = {
        'hip': 'red',    # 红色
        'tha': 'yellow',   # 黄色
        'white': 'green'     # 绿色
    }

    for region, positions in boxes_dict.items():
        for x, y in positions:
            rect = Rectangle((x, y), 200, 200, linewidth=2,
                           edgecolor=color_map[region], facecolor='none', alpha=0.8)
            ax.add_patch(rect)

    ax.set_title('Extracted Boxes (Black: hip, Yellow: tha, Red: white)')
    ax.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 1. 加载图像
    basepath = r'H:\Cross-scale_mouse_data\ST_data'
    tumor_path = Path(basepath) / 'tumor_8_brain'
    he_img = cv2.imread(str(tumor_path / 'Brain_8_aligned_HE.png'))
    mask_img = cv2.imread(str(tumor_path / 'Brain_8_HE_mask.png'))

    k = 40  # 每个区域抽取的box数量
    grid_factor = 1.2  # 网格密度因子（1.5-2.0之间调整）
    box_size = (200, 200)

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
        selected_positions = uniform_sample_positions(valid_positions, k, mask, box_size)
        boxes_dict[region] = selected_positions
        print(f"选择了 {len(selected_positions)} 个box")

    # 4. 可视化结果
    print("\n可视化结果...")
    save_path = 'box_location_m8.png'  # 设置保存图像名称
    visualize_boxes(he_img, boxes_dict)

    # 5. 保存坐标到Excel（按照hip、tha、white顺序）
    data = []
    region_order = ['hip', 'tha', 'white']  # 指定输出顺序
    for region in region_order:
        if region in boxes_dict:
            for x, y in boxes_dict[region]:
                data.append({'type': region, 'x': x, 'y': y})

    df = pd.DataFrame(data)
    excel_path = 'box_position_m8.xlsx'  # 设置保存位置表格名称
    df.to_excel(excel_path, index=False)
    print(f"结果已保存到: {excel_path}")

    # 6. 统计信息
    print("\n=== 统计信息 ===")
    for region in region_order:
        if region in boxes_dict:
            print(f"{region}: {len(boxes_dict[region])}个box")
    print(f"\n可视化图像: {save_path}")
    print(f"坐标数据: {excel_path}")
