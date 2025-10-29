import numpy as np
import open3d as o3d
import cv2
import os

from scipy.spatial.transform import Rotation

import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation

def generate_and_merge_point_cloud(rgb_input, depth_input, semantic_input,
                                  camera_intrinsic_params, camera_extrinsic_params,
                                   existing_rgb_pcd, existing_semantic_pcd):
    """
    生成RGB点云和语义点云，并分别将它们融合到已有的RGB点云和语义点云中

    参数:
        rgb_input: RGB图像(.png路径或numpy矩阵)
        depth_input: 深度图(.png路径或numpy矩阵)
        semantic_input: 语义掩码图(.png路径或numpy矩阵)
        camera_params: 相机参数向量 [fx, fy, cx, cy, k1, k2, p1, p2, k3, tx, ty, tz, qw, qx, qy, qz]
        existing_rgb_pcd_path: 已有的RGB点云文件路径
        existing_semantic_pcd_path: 已有的语义点云文件路径
        output_rgb_pcd_path: 输出的RGB点云文件路径
        output_semantic_pcd_path: 输出的语义点云文件路径
    """
    # 解析相机参数
    fx, fy, cx, cy = camera_intrinsic_params[0:4]  # 内参
    k1, k2, p1, p2, k3 = camera_intrinsic_params[4:9]  # 畸变系数
    translation = camera_extrinsic_params[0:3]  # 位移向量
    quaternion = camera_extrinsic_params[3:7]  # 四元数

    # 读取输入数据
    if isinstance(rgb_input, str):
        rgb_img = cv2.imread(rgb_input)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    else:
        rgb_img = rgb_input

    if isinstance(depth_input, str):
        depth_img = cv2.imread(depth_input, -1)  # 以原始格式读取深度图(16位)
        # 确保深度图是16位无符号整数类型
        if depth_img.dtype != np.uint16:
            depth_img = depth_img.astype(np.uint16)
    else:
        depth_img = depth_input

    if isinstance(semantic_input, str):
        semantic_img = cv2.imread(semantic_input)
        semantic_img = cv2.cvtColor(semantic_img, cv2.COLOR_BGR2RGB)
    else:
        semantic_img = semantic_input

    # # 深度图降噪处理
    # depth_img = cv2.medianBlur(depth_img, 5)  # 中值滤波
    # depth_img = cv2.GaussianBlur(depth_img, (5, 5), 0)  # 高斯滤波
    # 缩放深度图：将16位值转换为米（根据相机的深度单位调整）
    depth_float = depth_img.astype(np.float32) / 1000

    # 创建Open3D相机内参对象
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=rgb_img.shape[1],
        height=rgb_img.shape[0],
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

    # 从四元数创建旋转矩阵
    rotation_matrix = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()

    # 构建相机外参矩阵
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = translation

    # 创建Open3D深度图像对象
    depth_o3d = o3d.geometry.Image(depth_float)

    # 创建RGB点云
    rgb_o3d = o3d.geometry.Image(rgb_img)
    rgbd_image_rgb = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=1.0, convert_rgb_to_intensity=False
    )
    pcd_rgb = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_rgb, intrinsic, extrinsic_matrix
    )

    # 语义颜色映射
    semantic_colors = {
        'ditch': [127, 0, 0],
        'veg_bed': [0, 127, 0],
        'cem_grd': [127, 127, 0],
        'background': [0, 0, 0]
    }

    # 创建语义点云颜色
    semantic_cloud_colors = np.zeros_like(rgb_img, dtype=np.float64) / 255.0

    # 为每个像素分配语义颜色
    for semantic_name, color in semantic_colors.items():
        if semantic_name == 'background':
            continue  # 背景保持黑色
        mask = np.all(semantic_img == color, axis=2)
        semantic_cloud_colors[mask] = np.array(color) / 255.0

    # 创建语义点云
    semantic_o3d = o3d.geometry.Image((semantic_cloud_colors * 255).astype(np.uint8))
    rgbd_image_semantic = o3d.geometry.RGBDImage.create_from_color_and_depth(
        semantic_o3d, depth_o3d, depth_scale=1.0, depth_trunc=1.0, convert_rgb_to_intensity=False
    )
    pcd_semantic = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_semantic, intrinsic, extrinsic_matrix
    )

    # 合并RGB点云
    merged_rgb_pcd = existing_rgb_pcd + pcd_rgb

    # 合并语义点云
    merged_semantic_pcd = existing_semantic_pcd + pcd_semantic

    # 保存结果
    #o3d.io.write_point_cloud(output_rgb_pcd_path, merged_rgb_pcd)
    #o3d.io.write_point_cloud(output_semantic_pcd_path, merged_semantic_pcd)

    return merged_rgb_pcd, merged_semantic_pcd


"""时间戳列表读取函数"""


def list_loader(file_path):
    """
    从txt文件加载时间戳列表

    参数:
        file_path: 包含时间戳的txt文件路径

    返回:
        timestamps: 时间戳字符串数组
    """
    timestamps = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 分割每行，取第一列
                parts = line.strip().split()
                if parts:  # 确保行不为空
                    timestamps.append(parts[0])

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        return []
    except Exception as e:
        print(f"错误: 加载文件时发生异常: {e}")
        return []

    return timestamps

"""载入数据"""
def data_loader(current_timestamp, link_ct_path, link_path, traj_path, semantic_folder):
    """
    根据当前时间戳加载对应的数据文件路径和相机外参

    参数:
        current_timestamp: 当前时间戳字符串
        link_ct_path: link_CT.txt文件路径
        link_path: link.txt文件路径
        traj_path: traj.txt文件路径
        semantic_folder: 语义图存储文件夹路径

    返回:
        rgb_path: RGB图像路径
        depth_path: 深度图路径
        semantic_path: 语义图路径
        camera_params: 相机外参向量 [tx, ty, tz, qx, qy, qz, qw]
    """
    # 步骤1: 在link_CT.txt中查找当前时间戳对应的rgb时间戳
    rgb_timestamp = None
    try:
        with open(link_ct_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] == current_timestamp:
                    rgb_timestamp = parts[1]
                    break
    except FileNotFoundError:
        print(f"错误: 文件 '{link_ct_path}' 未找到")
        return None, None, None, None

    if rgb_timestamp is None:
        print(f"错误: 在 {link_ct_path} 中未找到时间戳 {current_timestamp} 的匹配项")
        return None, None, None, None

    # 步骤2: 在link.txt中查找rgb时间戳对应的rgb和深度图路径
    rgb_path = None
    depth_path = None
    try:
        with open(link_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0] == rgb_timestamp:
                    rgb_path = parts[1]
                    depth_path = parts[3]
                    break
    except FileNotFoundError:
        print(f"错误: 文件 '{link_path}' 未找到")
        return None, None, None, None

    if rgb_path is None or depth_path is None:
        print(f"错误: 在 {link_path} 中未找到时间戳 {rgb_timestamp} 的匹配项")
        return None, None, None, None

    # 步骤3: 构建语义图路径
    rgb_filename = os.path.basename(rgb_path)
    semantic_path = os.path.join(semantic_folder, rgb_filename)

    if not os.path.exists(semantic_path):
        print(f"警告: 语义图文件 '{semantic_path}' 不存在")
        # 可以选择返回None或尝试其他命名规则
        semantic_path = None

    # 步骤4: 在traj.txt中查找当前时间戳对应的相机外参
    camera_params = None
    try:
        with open(traj_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == current_timestamp:
                    # 提取相机外参: tx, ty, tz, qx, qy, qz, qw
                    camera_params = list(map(float, parts[1:8]))
                    break
    except FileNotFoundError:
        print(f"错误: 文件 '{traj_path}' 未找到")
        return None, None, None, None

    if camera_params is None:
        print(f"错误: 在 {traj_path} 中未找到时间戳 {current_timestamp} 的匹配项")
        return None, None, None, None

    return rgb_path, depth_path, semantic_path, camera_params


def create_initial_point_clouds(rgb_pcd_path, semantic_pcd_path):
    """
    创建初始点云文件，每个文件包含一个位于原点的点，并设置必要的属性

    参数:
        rgb_pcd_path: RGB点云文件路径
        semantic_pcd_path: 语义点云文件路径
        fused_pcd_path: 融合点云文件路径
    """
    # 创建RGB点云（包含颜色信息）
    rgb_pcd = o3d.geometry.PointCloud()
    rgb_pcd.points = o3d.utility.Vector3dVector([[0, 0, 0]])  # 原点位置
    rgb_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0]])  # 黑色(0,0,0)
    o3d.io.write_point_cloud(rgb_pcd_path, rgb_pcd)

    # 创建语义点云（包含颜色信息）
    semantic_pcd = o3d.geometry.PointCloud()
    semantic_pcd.points = o3d.utility.Vector3dVector([[0, 0, 0]])  # 原点位置
    semantic_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0]])  # 黑色(0,0,0)
    o3d.io.write_point_cloud(semantic_pcd_path, semantic_pcd)

    # 创建融合点云（包含颜色和语义字段）
    # 由于Open3D的legacy PointCloud不支持自定义字段，我们使用Tensor-based点云
    fused_pcd = o3d.t.geometry.PointCloud()
    fused_pcd.point["positions"] = o3d.core.Tensor([[0, 0, 0]], dtype=o3d.core.Dtype.Float32)
    fused_pcd.point["colors"] = o3d.core.Tensor([[0, 0, 0]], dtype=o3d.core.Dtype.Float32)
    fused_pcd.point["semantic"] = o3d.core.Tensor([[0]], dtype=o3d.core.Dtype.Int32)  # 语义标签为0(背景)

    print(f"初始化点云文件已创建:")
    print(f"RGB点云: {rgb_pcd_path}")
    print(f"语义点云: {semantic_pcd_path}")

    return rgb_pcd_path, semantic_pcd_path

"""主函数"""
def main():
    """
    主函数：处理时间戳列表，生成并合并点云
    """
    # 配置文件路径
    timestamp_file = "test\ORB3-KeyFrameTrajectory.txt"
    link_ct_file = "test\ORB3_KT_link.txt"
    link_file = "test\link.txt"
    traj_file = "test\ORB3-KeyFrameTrajectory.txt"
    semantic_folder = "test\segformer_outputs_raw"

    # 相机内参
    fx = 382.613
    fy = 382.613
    cx = 320.183
    cy = 236.455
    k1 = 0.0
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0
    k3 = 0.0

    # 构建相机内参向量
    camera_intrinsics = [fx, fy, cx, cy, k1, k2, p1, p2, k3]

    # 读取时间戳列表
    timestamps = list_loader(timestamp_file)
    if not timestamps:
        print("错误: 时间戳列表为空")
        return

    # 初始化三个点云文件
    rgb_pcd_path = "output/rgb.pcd"
    semantic_pcd_path = "output/semantic_visual.pcd"
    # fused_pcd_path = "output/with_semantic_field.pcd"
    output_folder = "output"

    rgb_pcd_path, semantic_pcd_path = create_initial_point_clouds(
        rgb_pcd_path, semantic_pcd_path
    )

    merged_semantic_pcd = o3d.io.read_point_cloud(semantic_pcd_path)
    merged_rgb_pcd = o3d.io.read_point_cloud(rgb_pcd_path)

    # 处理每个时间戳
    for i, timestamp in enumerate(timestamps):
        print(f"处理时间戳 {i + 1}/{len(timestamps)}: {timestamp}")

        # 加载数据
        rgb_path, depth_path, semantic_path, camera_extrinsics = data_loader(
            timestamp, link_ct_file, link_file, traj_file, semantic_folder
        )

        if not all([rgb_path, depth_path, semantic_path, camera_extrinsics]):
            print(f"警告: 时间戳 {timestamp} 的数据不完整，跳过")
            continue

        # 构建完整的相机参数向量 [内参 + 外参]
        # full_camera_params = camera_intrinsics + camera_extrinsics + [timestamp]

        # 生成并合并点云
        try:
            merged_rgb_pcd, merged_semantic_pcd = generate_and_merge_point_cloud(
                rgb_path, depth_path, semantic_path,
                camera_intrinsics,
                camera_extrinsics,
                merged_rgb_pcd, merged_semantic_pcd,  # 已有点云路径（融合点云）
            )

            print(f"时间戳 {timestamp} 处理完成")

        except Exception as e:
            print(f"错误: 处理时间戳 {timestamp} 时发生异常: {e}")
            continue

    # 保存结果
    o3d.io.write_point_cloud(rgb_pcd_path, merged_rgb_pcd)
    o3d.io.write_point_cloud(semantic_pcd_path, merged_semantic_pcd)
    print("所有时间戳处理完成")
    print(f"点云已保存至: {rgb_pcd_path}")


if __name__ == "__main__":
    main()
