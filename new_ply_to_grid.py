import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import csv
from scipy.spatial.transform import Rotation


def read_ply_file(file_path):
    """
    读取PLY文件并返回点云数据和颜色信息
    支持读取obj_dc_0到obj_dc_10字段，每个点只会有其中一个字段
    """
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.is_empty():
            print(f"成功读取PLY文件，包含 {len(pcd.points)} 个点")
            points = np.asarray(pcd.points)

            # 定义颜色映射表：0-10对应不同颜色
            color_mapping = {
                0: [0, 1, 0],  # 绿色
                1: [1, 0, 0],  # 红色
                2: [1, 1, 0],  # 黄色
                3: [0, 0, 1],  # 蓝色
                4: [1, 0, 1],  # 紫色
                5: [0, 1, 1],  # 青色
                6: [1, 0.5, 0],  # 橙色
                7: [0.5, 0, 0.5],  # 紫红色
                8: [0.5, 0.5, 0],  # 橄榄色
                9: [0, 0.5, 0.5],  # 蓝绿色
                10: [0.5, 0.5, 0.5]  # 灰色
            }

            # 初始化颜色数组为黑色
            colors = np.zeros((len(points), 3))

            # 尝试读取obj_dc_0到obj_dc_10字段
            try:
                # 读取PLY文件头获取属性信息
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # 查找顶点属性定义
                vertex_props = []
                in_vertex_section = False
                for line in lines:
                    line = line.strip()
                    if line.startswith('element vertex'):
                        in_vertex_section = True
                        continue
                    if line.startswith('element') and not line.startswith('element vertex'):
                        in_vertex_section = False
                        continue
                    if in_vertex_section and line.startswith('property'):
                        parts = line.split()
                        if len(parts) >= 3:
                            vertex_props.append(parts[2])

                # 检查哪些obj_dc字段存在
                dc_fields = []
                for i in range(11):  # 检查0到10
                    field_name = f"obj_dc_{i}"
                    if field_name in vertex_props:
                        dc_fields.append((i, field_name, vertex_props.index(field_name)))

                if not dc_fields:
                    print("警告: 未找到任何obj_dc_0到obj_dc_10字段，将使用黑色")
                    return points, colors

                print(f"找到 {len(dc_fields)} 个obj_dc字段: {[f[1] for f in dc_fields]}")

                # 读取数据部分
                data_lines = []
                in_data_section = False
                for line in lines:
                    if line.startswith('end_header'):
                        in_data_section = True
                        continue
                    if in_data_section:
                        data_lines.append(line.strip())

                # 为每个点分配颜色
                for point_idx, line in enumerate(data_lines):
                    if point_idx >= len(points):
                        break

                    parts = line.split()
                    found = False

                    # 检查每个点属于哪个obj_dc字段
                    for dc_id, field_name, field_idx in dc_fields:
                        if len(parts) > field_idx:
                            try:
                                # 假设存在该字段的点会有非零值
                                value = float(parts[field_idx])
                                if value > 0:  # 假设大于0表示该点属于这个类别
                                    colors[point_idx] = color_mapping[dc_id]
                                    found = True
                                    break
                            except ValueError:
                                continue

                    # 如果没有找到任何字段，保持黑色
                    if not found:
                        colors[point_idx] = [0, 0, 0]  # 黑色

                print("已从obj_dc字段生成颜色信息")
                return points, colors

            except Exception as e:
                print(f"读取obj_dc字段时出错: {e}")
                # 出错时返回黑色
                return points, colors

            return points, colors
        else:
            print("读取的PLY文件为空")
            return None, None
    except Exception as e:
        print(f"读取PLY文件时出错: {e}")
        return None, None


def rotate_points_to_plane(points, normal=[0, 0, 1]):
    """将点云旋转，使指定法向量的平面成为新的xy平面"""
    # 规范化法向量
    normal = np.array(normal) / np.linalg.norm(np.array(normal))

    # 如果法向量已经是z轴方向，则不需要旋转
    if np.allclose(normal, [0, 0, 1]):
        return points, np.eye(3)

    # 找到旋转轴 (当前法向量和z轴的叉积)
    rotation_axis = np.cross(normal, [0, 0, 1])
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # 计算旋转角度 (当前法向量和z轴的夹角)
    angle = np.arccos(np.dot(normal, [0, 0, 1]))

    # 构建旋转矩阵
    r = Rotation.from_rotvec(angle * rotation_axis)
    rotation_matrix = r.as_matrix()

    # 应用旋转
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    rotated_points = np.dot(centered_points, rotation_matrix.T) + centroid

    return rotated_points, rotation_matrix


def project_vector_to_plane(vector, normal):
    """将向量投影到指定法向量的平面上"""
    vector = np.array(vector)
    normal = np.array(normal) / np.linalg.norm(np.array(normal))

    # 计算投影
    projection = vector - np.dot(vector, normal) * normal
    return projection


def align_grid_to_direction(points, direction, normal):
    """
    旋转点云，使指定方向向量在指定平面上的投影成为新的X轴正方向
    """
    # 首先将点云旋转到指定平面
    points, R1 = rotate_points_to_plane(points, normal)

    # 计算方向向量在平面上的投影
    projected_direction = project_vector_to_plane(direction, [0, 0, 1])
    projected_direction = projected_direction / np.linalg.norm(projected_direction)

    # 计算从投影方向到x轴的旋转角度
    angle = np.arctan2(projected_direction[1], projected_direction[0])

    # 构建旋转矩阵
    R2 = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    # 应用第二次旋转
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    rotated_points = np.dot(centered_points, R2.T) + centroid

    # 组合两个旋转矩阵
    combined_rotation = np.dot(R2, R1)

    return rotated_points, combined_rotation


def create_occupancy_grid(points, colors=None, resolution=0.1, z_min=-2.0, z_max=2.0, threshold=0.5):
    """
    从点云创建包含颜色信息的占据栅格地图
    使用每个栅格中占比最多的点的颜色着色
    """
    if points is None or len(points) == 0:
        return None, None, None, None, None

    # 过滤Z轴范围外的点
    valid_indices = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    filtered_points = points[valid_indices]

    if colors is not None:
        filtered_colors = colors[valid_indices]
    else:
        filtered_colors = None

    if len(filtered_points) == 0:
        print("过滤后没有剩余点")
        return None, None, None, None, None

    # 计算边界
    x_min, y_min = np.min(filtered_points[:, :2], axis=0)
    x_max, y_max = np.max(filtered_points[:, :2], axis=0)

    # 计算栅格尺寸
    grid_size_x = int(np.ceil((x_max - x_min) / resolution))
    grid_size_y = int(np.ceil((y_max - y_min) / resolution))

    # 创建空的占据栅格和颜色栅格
    occupancy_grid = np.zeros((grid_size_y, grid_size_x))
    if filtered_colors is not None:
        color_grid = np.zeros((grid_size_y, grid_size_x, 3))
        # 存储每个栅格中的颜色计数
        color_counts = [[{} for _ in range(grid_size_x)] for _ in range(grid_size_y)]
    else:
        color_grid = None
        color_counts = None

    # 计算每个点所在的栅格索引
    indices = np.floor((filtered_points[:, :2] - [x_min, y_min]) / resolution).astype(int)

    # 过滤超出栅格范围的索引
    valid_indices = (indices[:, 0] >= 0) & (indices[:, 0] < grid_size_x) & \
                    (indices[:, 1] >= 0) & (indices[:, 1] < grid_size_y)
    indices = indices[valid_indices]
    filtered_points = filtered_points[valid_indices]

    if filtered_colors is not None:
        filtered_colors = filtered_colors[valid_indices]

    # 统计每个栅格中的点数和颜色计数
    points_per_cell = np.zeros((grid_size_y, grid_size_x))

    for idx, (i, j) in enumerate(indices):
        # 更新占据计数
        occupancy_grid[j, i] += 1
        points_per_cell[j, i] += 1

        # 统计颜色出现次数
        if filtered_colors is not None:
            color = tuple(filtered_colors[idx])
            if color in color_counts[j][i]:
                color_counts[j][i][color] += 1
            else:
                color_counts[j][i][color] = 1

    # 确定每个栅格中占比最多的颜色
    if color_grid is not None:
        for j in range(grid_size_y):
            for i in range(grid_size_x):
                if color_counts[j][i]:
                    # 找到出现次数最多的颜色
                    dominant_color = max(color_counts[j][i].items(), key=lambda x: x[1])[0]
                    color_grid[j, i] = dominant_color

    # 将点计数转换为占据概率
    max_count = np.max(occupancy_grid)
    if max_count > 0:
        occupancy_grid = occupancy_grid / max_count

    # 应用阈值进行二值化
    binary_grid = occupancy_grid >= threshold

    return binary_grid, occupancy_grid, color_grid, (x_min, y_min, x_max, y_max), resolution


def visualize_grid(grid, title="Occupancy Grid", save_path=None):
    """可视化占据栅格地图"""
    plt.figure(figsize=(10, 8))

    # 创建自定义颜色映射
    colors = [(0.8, 0.8, 0.8), (0.2, 0.2, 0.2)]  # 从浅灰到深灰
    cmap = LinearSegmentedColormap.from_list('OccupancyMap', colors, N=2)

    plt.imshow(grid, cmap=cmap, origin='lower')
    plt.title(title)
    plt.colorbar(label='Occupancy Probability')
    plt.grid(False)

    if save_path:
        plt.savefig(save_path)
        print(f"可视化结果已保存至 {save_path}")

    plt.show()


def visualize_color_grid(color_grid, title="Color Occupancy Grid", save_path=None):
    """可视化带颜色的占据栅格地图"""
    if color_grid is None:
        print("没有颜色信息可供可视化")
        return

    plt.figure(figsize=(10, 8))
    plt.imshow(color_grid, origin='lower')
    plt.title(title)
    plt.grid(False)

    # 添加颜色图例
    color_legend = [
        (0, "绿色 (obj_dc_0)"),
        (1, "红色 (obj_dc_1)"),
        (2, "黄色 (obj_dc_2)"),
        (3, "蓝色 (obj_dc_3)"),
        (4, "紫色 (obj_dc_4)"),
        (5, "青色 (obj_dc_5)"),
        (6, "橙色 (obj_dc_6)"),
        (7, "紫红色 (obj_dc_7)"),
        (8, "橄榄色 (obj_dc_8)"),
        (9, "蓝绿色 (obj_dc_9)"),
        (10, "灰色 (obj_dc_10)"),
        (-1, "黑色 (无字段)")
    ]

    # 在图像旁边添加图例
    legend_elements = []
    for idx, label in color_legend:
        if idx == -1:
            color = [0, 0, 0]  # 黑色
        else:
            color = [
                [0, 1, 0],  # 0: 绿色
                [1, 0, 0],  # 1: 红色
                [1, 1, 0],  # 2: 黄色
                [0, 0, 1],  # 3: 蓝色
                [1, 0, 1],  # 4: 紫色
                [0, 1, 1],  # 5: 青色
                [1, 0.5, 0],  # 6: 橙色
                [0.5, 0, 0.5],  # 7: 紫红色
                [0.5, 0.5, 0],  # 8: 橄榄色
                [0, 0.5, 0.5],  # 9: 蓝绿色
                [0.5, 0.5, 0.5]  # 10: 灰色
            ][idx]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color))

    plt.legend(legend_elements, [l[1] for l in color_legend],
               loc='center left', bbox_to_anchor=(1, 0.5))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"颜色可视化结果已保存至 {save_path}")

    plt.show()


def save_grid_to_csv(grid, file_path):
    """将栅格地图保存为CSV文件"""
    try:
        np.savetxt(file_path, grid, delimiter=',', fmt='%d')
        print(f"占据栅格地图已保存至 {file_path}")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")


def save_color_grid_to_csv(color_grid, file_path):
    """将颜色栅格地图保存为CSV文件"""
    if color_grid is None:
        print("没有颜色信息可供保存")
        return

    try:
        # 将RGB值转换为整数并保存
        color_grid_int = (color_grid * 255).astype(int)

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['i', 'j', 'r', 'g', 'b'])

            height, width = color_grid_int.shape[:2]
            for i in range(height):
                for j in range(width):
                    r, g, b = color_grid_int[i, j]
                    writer.writerow([i, j, r, g, b])

        print(f"颜色栅格地图已保存至 {file_path}")
    except Exception as e:
        print(f"保存颜色CSV文件时出错: {e}")


def main():
    # 直接在代码中设置参数
    config = {
        # 输入输出设置
        'input_file': 'weed_1000.ply',
        'output_occupancy': 'weed1000_occupancy_grid.csv',
        'output_color': 'weed1000_color_grid.csv',

        # 处理参数
        'resolution': 0.1,  # 栅格分辨率(米)
        'z_min': -2.0,  # Z轴最小值
        'z_max': 2.0,  # Z轴最大值
        'threshold': 0.5,  # 占据概率阈值

        # 平面法向量和方向向量
        'normal': [0, 1, 1],  # 新xy平面的法向量
        'direction': [0, 1, -1],  # 用于确定X轴方向的向量

        # 功能开关
        'visualize_occupancy': True,  # 可视化占据栅格
        'visualize_color': True,  # 可视化颜色栅格
        'save_occupancy_image': None,  # 保存占据栅格图像路径(None表示不保存)
        'save_color_image': None  # 保存颜色栅格图像路径(None表示不保存)
    }

    # 检查输入文件是否存在
    if not os.path.exists(config['input_file']):
        print(f"错误: 输入文件 {config['input_file']} 不存在")
        return

    # 读取PLY文件并获取颜色信息（自动处理obj_dc_0到obj_dc_10字段）
    points, colors = read_ply_file(config['input_file'])

    # 旋转点云使方向向量在平面上的投影成为新的X轴
    if config['direction'] != [1, 0, 0] or config['normal'] != [0, 0, 1]:
        print(f"将点云旋转，使法向量为 {config['normal']} 的平面成为新的xy平面")
        print(f"并使方向向量 {config['direction']} 在该平面上的投影成为新的X轴正方向")
        points, rotation_matrix = align_grid_to_direction(points, config['direction'], config['normal'])
        print(f"应用的旋转矩阵:\n{rotation_matrix}")

    # 创建占据栅格地图
    binary_grid, occupancy_grid, color_grid, bounds, resolution = create_occupancy_grid(
        points,
        colors=colors,
        resolution=config['resolution'],
        z_min=config['z_min'],
        z_max=config['z_max'],
        threshold=config['threshold']
    )

    if binary_grid is not None:
        # 保存栅格地图
        save_grid_to_csv(binary_grid, config['output_occupancy'])
        save_color_grid_to_csv(color_grid, config['output_color'])

        # 打印栅格地图信息
        x_min, y_min, x_max, y_max = bounds
        print(f"栅格地图尺寸: {binary_grid.shape[1]} x {binary_grid.shape[0]}")
        print(f"实际范围: X [{x_min:.2f}, {x_max:.2f}]m, Y [{y_min:.2f}, {y_max:.2f}]m")
        print(f"分辨率: {resolution}m")

        # 可视化
        if config['visualize_occupancy']:
            title = f"Occupancy Grid (Resolution: {resolution}m)"
            visualize_grid(binary_grid, title, config['save_occupancy_image'])

        if config['visualize_color'] and color_grid is not None:
            title = f"Color Occupancy Grid (Resolution: {resolution}m)"
            visualize_color_grid(color_grid, title, config['save_color_image'])


if __name__ == "__main__":
    main()