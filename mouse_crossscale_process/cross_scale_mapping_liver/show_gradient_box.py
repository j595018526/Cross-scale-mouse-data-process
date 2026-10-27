"""
从肿瘤内部区域中心点出发，向右滑动采样400x400的box
修改为支持显示多个指定的box（不保存文件）
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==================== 参数设置 ====================
basepath = r'H:\Cross-scale_mouse_data\ST_data'
tumor_path = Path(basepath) / 'tumor_8_liver'
he_img = cv2.imread(str(tumor_path / 'Liver_8_aligned_HE.png'))
mask_img = cv2.imread(str(tumor_path / 'Liver_8_HE_mask.png'))

# 可调节参数
box_size = 400  # box大小
step_size = 100  # 滑动步长
num_steps = 40 - 1  # 滑动步数
target_box_indices = [14]  # 想要显示的第几个box（从1开始计数），支持列表

# ==================== 主程序 ====================
# 1. 提取肿瘤内部mask
tumor_color = np.array([36, 28, 237])  # BGR格式
tumor_mask = cv2.inRange(mask_img, tumor_color, tumor_color)

# 2. 计算几何中心点
coords = np.where(tumor_mask > 0)
if len(coords[0]) == 0:
    raise ValueError("未找到肿瘤内部区域！")

center_y = int(np.mean(coords[0]))
center_x = int(np.mean(coords[1]))
print(f"肿瘤内部区域中心点: ({center_x}, {center_y})")

# 3. 生成所有box位置（从中心点开始，向右滑动）
boxes = []
valid_indices = []  # 记录有效box的序号
for i in range(num_steps + 1):
    x = center_x - box_size // 2 + i * step_size
    y = center_y - box_size // 2

    # 确保box在图像范围内
    if x >= 0 and x + box_size <= he_img.shape[1] and y >= 0 and y + box_size <= he_img.shape[0]:
        boxes.append((x, y))
        valid_indices.append(i + 1)  # 记录有效序号（从1开始）

print(f"共生成 {len(boxes)} 个有效box")

# 4. 处理target_box_indices（确保是列表）
if not isinstance(target_box_indices, list):
    target_box_indices = [target_box_indices]

# 过滤出有效的box序号
valid_targets = [idx for idx in target_box_indices if idx in valid_indices]
if not valid_targets:
    print(f"警告: 指定的box {target_box_indices} 均不存在或超出图像边界！")
    print(f"可用的box序号: {valid_indices[:10]}... (共{len(valid_indices)}个)")
    # 使用第一个有效box作为替代
    valid_targets = [valid_indices[0]]
    print(f"将显示第{valid_targets[0]}个box作为替代")

# 获取目标box的坐标
target_boxes = []
for idx in valid_targets:
    target_idx = valid_indices.index(idx)
    target_boxes.append((idx, boxes[target_idx][0], boxes[target_idx][1]))

# ==================== 打印目标Box信息 ====================
print(f"\n将显示 {len(target_boxes)} 个box:")
for idx, x, y in target_boxes:
    print(f"  Box #{idx}: 左上角坐标 ({x}, {y})")

# 5. 可视化（显示所有目标box）
cell_red = '#D62728'  # Cell子刊经典红色
box_color = cell_red  # 统一使用该颜色

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
# 显示原始病理图像
ax.imshow(cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB))

# 绘制所有目标box（统一颜色）
for idx, x, y in target_boxes:
    rect = Rectangle((x, y), box_size, box_size,
                     linewidth=3,
                     edgecolor=box_color,
                     facecolor='none')
    ax.add_patch(rect)

# 标记肿瘤中心点（使用黑色）
ax.scatter(center_x, center_y,
          c='cyan',
          s=220,
          marker='+',
          linewidths=5,
          label='Tumor center')

ax.axis('off')
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout(pad=0)
# 显示图像（不保存）
plt.show()

print(f"\n显示完成！")
print(f"参数: 步长={step_size}px, 步数={num_steps}, box大小={box_size}px")
print(f"显示了 {len(target_boxes)} 个box: {[idx for idx, _, _ in target_boxes]}")
