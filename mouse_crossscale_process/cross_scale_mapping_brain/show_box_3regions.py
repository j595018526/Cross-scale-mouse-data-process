import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# 1. 加载图像和表格
basepath = r'H:\Cross-scale_mouse_data\ST_data'
tumor_path = Path(basepath) / 'tumor_8_brain'
he_img = cv2.imread(str(tumor_path / 'Brain_8_aligned_HE.png'))
df = pd.read_excel('./brain_data/box_position_m8.xlsx')

# 2. 在图像上显示box
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.imshow(cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB))

# 颜色映射
color_map = {'hip': 'red', 'tha': 'yellow', 'white': 'green'}

# 绘制所有box
for _, row in df.iterrows():
    rect = Rectangle((row['x'], row['y']), 200, 200,
                     linewidth=2, edgecolor=color_map[row['type']],
                     facecolor='none', alpha=0.8)
    ax.add_patch(rect)

ax.set_title('Box Positions (red: hip, yellow: tha, green: white)')
ax.axis('off')
plt.show()

print(f"共显示 {len(df)} 个box")
