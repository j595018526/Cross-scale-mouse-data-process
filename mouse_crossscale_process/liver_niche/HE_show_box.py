
#################################根据box顶点坐标，在原病理图像中用不同颜色显示不同类型的box#####################################
import cv2
import pandas as pd

# 读取数据
img = cv2.imread('D:/mouse3.png')
df = pd.read_excel('C:/Users/j5950/Desktop/模态映射-论文撰写/小鼠肝-生态位/box_locations.xlsx', sheet_name='mouse3')

# 颜色映射 (BGR格式)
colors = {'niche': (0, 0, 255), 'tumor': (0, 255, 255), 'normal': (0, 255, 0)}

# 绘制box
for _, row in df.iterrows():
    if row['type'] in colors:
        cv2.rectangle(img, (row['x'], row['y']), (row['x']+400, row['y']+400), colors[row['type']], 18)

# 保存为PNG图像
cv2.imwrite('D:/result3.png', img)
print("图像已保存为 result.png")
