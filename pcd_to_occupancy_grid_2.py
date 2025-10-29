import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import csv
from scipy.spatial.transform import Rotation
import time


def read_ply_file(file_path):
    """读取PLY文件并返回点云数据和颜色信息"""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.is_empty():
            print(f"成功读取PLY文件，包含 {len(pcd.points)} 个点")
            points = np.asarray(pcd.points)

            # 检查是否有颜色信息
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                print("点云包含颜色信息")
                return points, colors
            else:
                print("点云不包含颜色信息，将使用默认颜色")
                return points, None
        else:
            print("读取的PLY文件为空")
            return None, None
    except Exception as e:
        print(f"读取PLY文件时出错: {e}")
        return None, None


#统计滤波去噪
def statistical_outlier_removal(points, colors=None, nb_neighbors=20, std_ratio=2.0):
    """
    使用统计滤波去除离群点

    参数:
    points: 点云数据
    colors: 颜色数据（可选）
    nb_neighbors: 考虑的邻近点数量
    std_ratio: 标准差比率，值越小过滤越严格

    返回:
    过滤后的点和颜色
    """
    try:
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # 执行统计滤波
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # 获取过滤后的点
        filtered_points = np.asarray(cl.points)

        if colors is not None:
            filtered_colors = np.asarray(cl.colors)
        else:
            filtered_colors = None

        print(f"统计滤波: 原始点数 {len(points)} → 过滤后 {len(filtered_points)}")

        return filtered_points, filtered_colors

    except Exception as e:
        print(f"统计滤波出错: {e}")
        return points, colors

def rotate_points_to_plane(points, normal=[0, 0, 1]):
    """
    将点云旋转，使指定法向量的平面成为新的xy平面

    参数:
    points: 点云数据，numpy数组，形状为(N, 3)
    normal: 指定平面的法向量，默认是z轴方向[0,0,1]

    返回:
    rotated_points: 旋转后的点云
    rotation_matrix: 旋转矩阵
    """
    # 规范化法向量
    normal = np.array(normal) / np.linalg.norm(np.array(normal))

    # 计算从当前法向量旋转到z轴的旋转矩阵
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


def create_occupancy_grid(points, colors=None, resolution=0.1, z_min=-2.0, z_max=2.0, threshold=0.5):
    """
    从点云创建包含颜色信息的占据栅格地图

    参数:
    points: 点云数据，numpy数组，形状为(N, 3)
    colors: 颜色数据，numpy数组，形状为(N, 3)，取值范围[0,1]
    resolution: 栅格分辨率(米)
    z_min, z_max: 考虑的Z轴范围
    threshold: 占据概率阈值，高于此值的栅格被标记为占据
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
    else:
        color_grid = None

    # 计算每个点所在的栅格索引
    indices = np.floor((filtered_points[:, :2] - [x_min, y_min]) / resolution).astype(int)

    # 过滤超出栅格范围的索引
    valid_indices = (indices[:, 0] >= 0) & (indices[:, 0] < grid_size_x) & \
                    (indices[:, 1] >= 0) & (indices[:, 1] < grid_size_y)
    indices = indices[valid_indices]
    filtered_points = filtered_points[valid_indices]

    if filtered_colors is not None:
        filtered_colors = filtered_colors[valid_indices]

    # 统计每个栅格中的点数并计算平均颜色
    points_per_cell = np.zeros((grid_size_y, grid_size_x))

    for idx, (i, j) in enumerate(indices):
        # 更新占据计数
        occupancy_grid[j, i] += 1
        points_per_cell[j, i] += 1

        # 更新颜色
        if filtered_colors is not None:
            color_grid[j, i] += filtered_colors[idx]

    # 计算平均颜色
    if color_grid is not None:
        valid_cells = points_per_cell > 0
        color_grid[valid_cells] /= points_per_cell[valid_cells, np.newaxis]

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

    if save_path:
        plt.savefig(save_path)
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


def process_point_cloud(input_file,
                        output_file='occupancy_grid.csv',
                        color_output_file='color_grid.csv',
                        resolution=0.1,
                        z_min=-2.0,
                        z_max=2.0,
                        threshold=0.5,
                        normal=[0, 0, 1],
                        visualize=False,
                        vis_save=None,
                        color_vis_save=None,
                        filter_params=None):
    """处理点云并生成占据栅格地图的主函数"""
    # 读取PLY文件
    points, colors = read_ply_file(input_file)

    #计时
    start = time.perf_counter()

    #滤波
    if points is not None:
        # 默认参数或自定义参数
        nb_neighbors = filter_params.get('nb_neighbors', 20) if filter_params else 20
        std_ratio = filter_params.get('std_ratio', 2.0) if filter_params else 2.0
        points, colors = statistical_outlier_removal(points, colors, nb_neighbors, std_ratio)

    # 如果指定了法向量，旋转点云
    if normal != [0, 0, 1]:
        print(f"将点云旋转，使法向量为 {normal} 的平面成为新的xy平面")
        points, rotation_matrix = rotate_points_to_plane(points, normal)
        print(f"应用的旋转矩阵:\n{rotation_matrix}")

    # 创建占据栅格地图
    binary_grid, occupancy_grid, color_grid, bounds, resolution = create_occupancy_grid(
        points,
        colors=colors,
        resolution=resolution,
        z_min=z_min,
        z_max=z_max,
        threshold=threshold
    )

    # 记录结束时间
    end = time.perf_counter()

    # 计算并打印运行时间
    elapsed = end - start
    print(f"代码运行耗时: {elapsed:.6f} 秒")
    print(f"代码运行耗时: {elapsed * 1000:.3f} 毫秒")

    if binary_grid is not None:
        # 保存栅格地图
        save_grid_to_csv(binary_grid, output_file)
        save_color_grid_to_csv(color_grid, color_output_file)

        # 打印栅格地图信息
        x_min, y_min, x_max, y_max = bounds
        print(f"栅格地图尺寸: {binary_grid.shape[1]} x {binary_grid.shape[0]}")
        print(f"实际范围: X [{x_min:.2f}, {x_max:.2f}]m, Y [{y_min:.2f}, {y_max:.2f}]m")
        print(f"分辨率: {resolution}m")

        # 可视化
        if visualize:
            if vis_save:
                title = f"Occupancy Grid (Resolution: {resolution}m)"
                visualize_grid(binary_grid, title, vis_save)
            if color_vis_save or colors is not None:
                title = f"Color Occupancy Grid (Resolution: {resolution}m)"
                visualize_color_grid(color_grid, title, color_vis_save)


if __name__ == "__main__":
    # 直接在代码中设置参数
    input_file = "dapeng_s_pointcloud/dapeng_3/semantic_fused.ply"
    output_file = "dapeng3_occupancy_grid.csv"
    color_output_file = "dapeng3_color_grid.csv"
    resolution = 0.02  # 栅格分辨率(米)
    z_min = -2.0  # Z轴最小值(米)
    z_max = 2.0  # Z轴最大值(米)
    threshold = 0.3  # 占据概率阈值
    normal = [0.0283768, -0.841478, -0.539546]  # 指定新xy平面的法向量

    # 滤波参数
    filter_params = {
        'nb_neighbors': 10,  # 统计滤波参数
        'std_ratio': 0.5  # 统计滤波参数
    }

    # 是否可视化结果
    visualize = True
    vis_save = "occupancy.png"
    color_vis_save = "semantic_color.png"

    # 调用处理函数
    process_point_cloud(
        input_file=input_file,
        output_file=output_file,
        color_output_file=color_output_file,
        resolution=resolution,
        z_min=z_min,
        z_max=z_max,
        threshold=threshold,
        normal=normal,
        visualize=visualize,
        vis_save=vis_save,
        color_vis_save=color_vis_save,
        filter_params=filter_params
    )