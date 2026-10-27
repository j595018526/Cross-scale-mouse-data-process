# """
# 从两个指定方向取连续box，只显示指定的box编号
# """
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# # ==================== 参数设置 ====================
# basepath = r'H:\Cross-scale_mouse_data\ST_data'
# tumor_path = Path(basepath) / 'tumor_8_brain'
# he_img = cv2.imread(str(tumor_path / 'Brain_8_aligned_HE.png'))
#
# # box参数
# box_size = 200
# overlap = 100
# step = box_size - overlap  # 步长 = 100
#
# # 两个方向的起点和终点 (x, y)
# directions = [
#     ((4260, 115), (5620, 3415)),  # 方向1
#     ((7700, 105), (5620, 3415))   # 方向2
# ]
#
# # 指定要显示的box编号（从1开始计数）- 这些编号是翻转后的编号
# box_show1 = [9, 14, 32]  # 方向1（翻转后的编号）
# box_show2 = [11, 17, 32]  # 方向2（翻转后的编号）
#
# # 两组box使用不同颜色
# color_group1 = '#D62728'  # 经典红（深红）————对应方向1
# color_group2 = '#E8927C'  # 浅珊瑚红（浅红）————对应方向2
#
# # ==================== 主程序 ====================
# fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# ax.imshow(cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB))
#
# h, w = he_img.shape[:2]
#
# # 存储所有要显示的box及其对应的颜色和编号
# boxes_to_show = []
#
# for dir_idx, (start, end) in enumerate(directions):
#     x1, y1 = start
#     x2, y2 = end
#
#     # 选择当前方向的box编号列表和颜色
#     show_indices = box_show1 if dir_idx == 0 else box_show2
#     box_color = color_group1 if dir_idx == 0 else color_group2
#
#     # 计算方向向量和总距离
#     dx, dy = x2 - x1, y2 - y1
#     total_dist = np.sqrt(dx ** 2 + dy ** 2)
#
#     # 计算所有可能的box中心点
#     num_boxes = int(total_dist / step) + 1
#
#     # 生成并存储所有有效box的坐标
#     all_valid_boxes = []
#     for i in range(num_boxes):
#         t = i * step / total_dist
#         if t > 1:
#             break
#         cx = x1 + t * dx
#         cy = y1 + t * dy
#
#         x = int(cx - box_size / 2)
#         y = int(cy - box_size / 2)
#
#         if (x >= 0 and y >= 0 and
#                 x + box_size <= w and
#                 y + box_size <= h):
#             all_valid_boxes.append((x, y))
#
#     # 如果是方向2，抛弃最后一个box
#     if dir_idx == 1 and len(all_valid_boxes) > 0:
#         all_valid_boxes = all_valid_boxes[:-1]
#
#     # 翻转编号还原
#     total_boxes = len(all_valid_boxes)
#
#     # 提取指定的box（编号从1开始）
#     for flipped_idx in show_indices:
#         original_idx = total_boxes - flipped_idx + 1
#
#         if 1 <= original_idx <= total_boxes:
#             x, y = all_valid_boxes[original_idx - 1]
#             boxes_to_show.append((x, y, box_color, flipped_idx))
#
#             print(f"方向{dir_idx + 1}: 翻转编号 #{flipped_idx} -> 原始编号 #{original_idx} 位置 ({x}, {y})")
#         else:
#             print(f"方向{dir_idx + 1}: 翻转编号 #{flipped_idx} 超出范围 (总box数: {total_boxes})")
#
# # 绘制所有选中的box
# for x, y, color, idx in boxes_to_show:
#     # 绘制矩形边框
#     rect = plt.Rectangle((x, y), box_size, box_size, linewidth=3, edgecolor=color, facecolor='none', alpha=1.0)
#     ax.add_patch(rect)
#
#     # 在box中心添加编号（纯黑色文字，无背景）
#     center_x = x + box_size // 2
#     center_y = y + box_size // 2
#     ax.text(center_x, center_y, f"{idx}", color='black', fontsize=14, fontweight='bold', ha='center', va='center')
#
# ax.axis('off')
# plt.tight_layout(pad=0)
# plt.show()
#
# print(f"\n完成! 方向1显示了 {len(box_show1)} 个box, 方向2显示了 {len(box_show2)} 个box")



"""
从两个指定方向取连续box，只显示指定的box编号。病理图像翻转后显示
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 参数设置 ====================
basepath = r'H:\Cross-scale_mouse_data\ST_data'
tumor_path = Path(basepath) / 'tumor_3_brain'
he_img = cv2.imread(str(tumor_path / 'Brain_3_aligned_HE.png'))

# box参数
box_size = 200
overlap = 100
step = box_size - overlap  # 步长 = 100

# 两个方向的起点和终点 (x, y)
directions = [
    ((3920, 14), (4660, 4120)),  # 方向1
    ((7114, 14), (5860, 4134))   # 方向2
]

# 指定要显示的box编号（从1开始计数）- 这些编号是翻转后的编号
box_show1 = [9, 16, 36]  # 方向1（翻转后的编号）
box_show2 = [8, 16, 36]  # 方向2（翻转后的编号）

# 两组box使用不同颜色
color_group1 = '#D62728'  # 经典红（深红）————对应方向1
color_group2 = '#E8927C'  # 浅珊瑚红（浅红）————对应方向2

# ==================== 主程序 ====================
# ====== 修改开始：对图像进行上下翻转 ======
he_img_flipped = cv2.flip(he_img, 0)  # 0表示上下翻转
h, w = he_img_flipped.shape[:2]
# ====== 修改结束 ======

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# ====== 修改开始：显示翻转后的图像 ======
ax.imshow(cv2.cvtColor(he_img_flipped, cv2.COLOR_BGR2RGB))
# ====== 修改结束 ======

# 存储所有要显示的box及其对应的颜色和编号
boxes_to_show = []

for dir_idx, (start, end) in enumerate(directions):
    x1, y1 = start
    x2, y2 = end

    # 选择当前方向的box编号列表和颜色
    show_indices = box_show1 if dir_idx == 0 else box_show2
    box_color = color_group1 if dir_idx == 0 else color_group2

    # 计算方向向量和总距离
    dx, dy = x2 - x1, y2 - y1
    total_dist = np.sqrt(dx ** 2 + dy ** 2)

    # 计算所有可能的box中心点
    num_boxes = int(total_dist / step) + 1

    # 生成并存储所有有效box的坐标
    all_valid_boxes = []
    for i in range(num_boxes):
        t = i * step / total_dist
        if t > 1:
            break
        cx = x1 + t * dx
        cy = y1 + t * dy

        x = int(cx - box_size / 2)
        y = int(cy - box_size / 2)

        if (x >= 0 and y >= 0 and
                x + box_size <= w and
                y + box_size <= h):
            all_valid_boxes.append((x, y))

    # 如果是方向2，抛弃最后一个box
    if dir_idx == 1 and len(all_valid_boxes) > 0:
        all_valid_boxes = all_valid_boxes[:-1]

    # 翻转编号还原
    total_boxes = len(all_valid_boxes)

    # 提取指定的box（编号从1开始）
    for flipped_idx in show_indices:
        original_idx = total_boxes - flipped_idx + 1

        if 1 <= original_idx <= total_boxes:
            x, y = all_valid_boxes[original_idx - 1]
            # ====== 修改开始：对box坐标进行上下翻转 ======
            y_flipped = h - y - box_size  # 上下翻转后的y坐标
            boxes_to_show.append((x, y_flipped, box_color, flipped_idx))
            # ====== 修改结束 ======

            print(f"方向{dir_idx + 1}: 翻转编号 #{flipped_idx} -> 原始编号 #{original_idx} 位置 ({x}, {y})")
        else:
            print(f"方向{dir_idx + 1}: 翻转编号 #{flipped_idx} 超出范围 (总box数: {total_boxes})")

# 绘制所有选中的box
for x, y, color, idx in boxes_to_show:
    # 绘制矩形边框
    rect = plt.Rectangle((x, y), box_size, box_size, linewidth=3, edgecolor=color, facecolor='none', alpha=1.0)
    ax.add_patch(rect)

    # 在box中心添加编号（纯黑色文字，无背景）
    center_x = x + box_size // 2
    center_y = y + box_size // 2
    ax.text(center_x, center_y, f"{idx}", color='black', fontsize=14, fontweight='bold', ha='center', va='center')

ax.axis('off')
plt.tight_layout(pad=0)
plt.show()

print(f"\n完成! 方向1显示了 {len(box_show1)} 个box, 方向2显示了 {len(box_show2)} 个box")
