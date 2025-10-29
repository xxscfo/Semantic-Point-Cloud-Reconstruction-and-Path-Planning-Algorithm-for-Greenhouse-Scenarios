import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('dapeng_1/grid_map.csv')

# 获取唯一的x和y坐标值并排序
unique_x = sorted(df['x'].unique())
unique_y = sorted(df['y'].unique())

# 创建映射字典：坐标值 -> 位置序号
x_mapping = {coord: idx + 1 for idx, coord in enumerate(unique_x)}
y_mapping = {coord: idx + 1 for idx, coord in enumerate(unique_y)}

# 添加位置序号列
df['x_index'] = df['x'].map(x_mapping)
df['y_index'] = df['y'].map(y_mapping)

# 重新排列列的顺序
df = df[['x', 'y', 'x_index', 'y_index', 'R', 'G', 'B']]

# 保存结果到新文件（可选）
df.to_csv('dapeng_1/rgb_grid_map.csv', index=False)

# 显示前几行结果
print("转换后的前10行数据：")
print(df.head(10))

# 显示坐标映射关系
print(f"\nX坐标映射: {x_mapping}")
print(f"Y坐标映射: {y_mapping}")
print(f"\n网格大小: {len(unique_x)} × {len(unique_y)}")