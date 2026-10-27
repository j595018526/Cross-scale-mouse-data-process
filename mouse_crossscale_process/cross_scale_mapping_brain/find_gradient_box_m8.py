"""
从两个指定方向取连续box，绘制在原病理图像上
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ==================== 参数设置 ====================
basepath = r'H:\Cross-scale_mouse_data\ST_data'
tumor_path = Path(basepath) / 'tumor_8_brain'
he_img = cv2.imread(str(tumor_path / 'Brain_8_aligned_HE.png'))

# box参数
box_size = 200
overlap = 100
step = box_size - overlap  # 步长 = 100

# 两个方向的起点和终点 (x, y)
directions = [
    ((4260, 115), (5620, 3415)),  # 方向1
    ((7700, 105), (5620, 3415))   # 方向2
]

# Cell子刊风格红色色系（两个方向使用不同深浅的红色）
color_group1 = '#D62728'  # 经典红（深红）
color_group2 = '#E8927C'  # 浅珊瑚红（浅红）

# ==================== 主程序 ====================
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.imshow(cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB))

h, w = he_img.shape[:2]  # 获取图像尺寸

# 存储两个方向的box坐标
all_boxes = []

for dir_idx, (start, end) in enumerate(directions):
    x1, y1 = start
    x2, y2 = end

    # 计算方向向量和总距离
    dx, dy = x2 - x1, y2 - y1
    total_dist = np.sqrt(dx ** 2 + dy ** 2)

    # 计算box数量（从起点到终点，步长step）
    num_boxes = int(total_dist / step) + 1

    # 生成box中心点并绘制
    valid_boxes = 0
    box_positions = []  # 存储所有有效box的位置

    for i in range(num_boxes):
        t = i * step / total_dist
        if t > 1:
            break
        cx = x1 + t * dx
        cy = y1 + t * dy

        # box左上角坐标
        x = int(cx - box_size / 2)
        y = int(cy - box_size / 2)

        # 检查box是否完全在图像范围内
        if (x >= 0 and y >= 0 and
            x + box_size <= w and
            y + box_size <= h):
            box_positions.append((x, y))
            valid_boxes += 1

    # 存储当前方向的box坐标（不再抛弃最后一个box）
    all_boxes.append(box_positions)

    # 选择当前方向的颜色
    current_color = color_group1 if dir_idx == 0 else color_group2
    group_label = f"Group {dir_idx + 1}"

    # 绘制所有有效box
    for x, y in box_positions:
        rect = plt.Rectangle((x, y), box_size, box_size,
                             linewidth=2, edgecolor=current_color,
                             facecolor='none', alpha=0.8)
        ax.add_patch(rect)

    print(f"方向 ({x1},{y1})->({x2},{y2}): 生成 {valid_boxes} 个有效box (超出边界的已删除)")

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='none', edgecolor=color_group1, label='Group 1', linewidth=2),
    Patch(facecolor='none', edgecolor=color_group2, label='Group 2', linewidth=2)
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
          frameon=True, fancybox=True, shadow=True, edgecolor='lightgray')

ax.axis('off')
plt.tight_layout(pad=0)
plt.show()

print(f"\n完成! box尺寸: {box_size}px, 重叠: {overlap}px, 步长: {step}px")

# ==================== 保存box坐标到xlsx ====================
output_path = 'gradient_box_location_m8.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for idx, boxes in enumerate(all_boxes, start=1):
        df = pd.DataFrame(boxes, columns=['x', 'y'])
        df.insert(0, 'ID', range(1, len(df) + 1))  # 添加ID列从1开始
        # 保存到对应的sheet
        sheet_name = f'box_group{idx}'
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"已保存 {sheet_name}: {len(df)} 个box坐标")

print(f"\n坐标已保存至: {output_path}")
