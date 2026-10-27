# import pandas as pd
# from PIL import Image, ImageDraw
# import numpy as np
# import re
#
# # 文件路径
# csv_path = 'H:/Cross-scale_mouse_data/ST_data/Niche_spot_coordinates.csv'
# he_path = 'H:/Cross-scale_mouse_data/ST_data/tumor_3_brain/Brain_3_aligned_HE.png'
# output_path = 'D:/PythonCode/mouse_process/image_brain/cell_distribution'
#
# # 读取HE图像
# img = Image.open(he_path)
# img_array = np.array(img)
# img_height, img_width = img_array.shape[:2]
#
# # 读取CSV并筛选
# df = pd.read_csv(csv_path)
# filtered = df[df['sample'] == 'Brain_3']
#
# # 坐标转换参数
# scale_ratio_x = 56500 / img_width
# scale_ratio_y = 20434 / img_height
#
# # 获取dominant_celltype的不重复元素
# cell_types = filtered['dominant_celltype'].unique()
#
#
# # 定义文件名安全函数
# def safe_filename(name):
#     # 替换Windows文件名中的非法字符
#     return re.sub(r'[<>:"/\\|?*]', '_', name)
#
#
# # 为每个celltype绘制图像
# for cell_type in cell_types:
#     # 筛选当前celltype的数据
#     subset = filtered[filtered['dominant_celltype'] == cell_type]
#
#     # 转换坐标
#     x_coords = subset['x'].values / scale_ratio_x
#     y_coords = subset['y'].values / scale_ratio_y
#
#     # 在图像上绘制红色点
#     img_overlay = img.convert('RGBA')
#     draw = ImageDraw.Draw(img_overlay)
#
#     point_color = (255, 0, 0, 255)
#     point_radius = 10
#
#     for x, y in zip(x_coords, y_coords):
#         left_up = (x - point_radius, img_height - y - point_radius)
#         right_down = (x + point_radius, img_height - y + point_radius)
#         draw.ellipse([left_up, right_down], fill=point_color)
#
#     # 使用安全文件名保存
#     safe_name = safe_filename(cell_type)
#     save_path = f'{output_path}/{safe_name}.png'
#     img_overlay.save(save_path)
#     print(f"已保存至: {save_path}，共绘制{len(x_coords)}个点")



import pandas as pd
from PIL import Image, ImageDraw
import numpy as np

# 文件路径
csv_path = 'H:/Cross-scale_mouse_data/ST_data/Niche_spot_coordinates.csv'
he_path = 'H:/Cross-scale_mouse_data/ST_data/tumor_3_brain/Brain_3_aligned_HE.png'
output_path = 'D:/PythonCode/mouse_process/image_brain/cell_distribution'

# 读取HE图像
img = Image.open(he_path)
img_array = np.array(img)
img_height, img_width = img_array.shape[:2]

# 读取CSV并筛选
df = pd.read_csv(csv_path)
filtered = df[df['sample'] == 'Brain_3']

# 坐标转换参数
scale_ratio_x = 56500 / img_width
scale_ratio_y = 20434 / img_height

# 转换所有坐标
x_coords = filtered['x'].values / scale_ratio_x
y_coords = filtered['y'].values / scale_ratio_y

# 在图像上绘制红色点
img_overlay = img.convert('RGBA')
draw = ImageDraw.Draw(img_overlay)

point_color = (255, 0, 0, 255)
point_radius = 10

for x, y in zip(x_coords, y_coords):
    left_up = (x - point_radius, img_height - y - point_radius)
    right_down = (x + point_radius, img_height - y + point_radius)
    draw.ellipse([left_up, right_down], fill=point_color)

# 保存结果
save_path = f'{output_path}/all_cells.png'
img_overlay.save(save_path)
print(f"已保存至: {save_path}，共绘制{len(x_coords)}个点")
