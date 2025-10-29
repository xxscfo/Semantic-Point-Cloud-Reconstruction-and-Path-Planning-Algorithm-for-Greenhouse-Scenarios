import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import heapq
import math


class Node:
    def __init__(self, x, y, g_cost=float('inf'), h_cost=0, parent=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost  # 从起点到当前节点的成本
        self.h_cost = h_cost  # 启发式成本（到终点的估计成本）
        self.parent = parent  # 父节点

    def f_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost() < other.f_cost()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class AStarPathPlanner:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.grid = self.create_grid()
        self.min_x = int(min(self.data['x']))
        self.min_y = int(min(self.data['y']))
        self.max_x = int(max(self.data['x']))
        self.max_y = int(max(self.data['y']))
        self.width = self.max_x - self.min_x + 1
        self.height = self.max_y - self.min_y + 1

    def create_grid(self):
        # 创建网格，存储每个格点的颜色信息
        min_x = int(min(self.data['x']))
        min_y = int(min(self.data['y']))
        max_x = int(max(self.data['x']))
        max_y = int(max(self.data['y']))

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # 初始化网格为None
        grid = [[None for _ in range(height)] for _ in range(width)]

        # 填充网格
        for _, row in self.data.iterrows():
            x = int(row['x']) - min_x
            y = int(row['y']) - min_y
            r, g, b = int(row['R']), int(row['G']), int(row['B'])
            grid[x][y] = (r, g, b)

        return grid

    def is_obstacle(self, x, y):
        # 检查坐标是否在网格范围内
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        # 获取颜色信息
        color = self.grid[x][y]
        if color is None:
            return True

        # 红色或黑色是障碍物
        r, g, b = color
        return (r == 127 and g == 0 and b == 0) or (r == 0 and g == 0 and b == 0)

    def is_valid_position(self, x, y, is_diagonal=False):
        # 检查车辆中心点位置是否有效
        if is_diagonal:
            # 斜向移动，检查4×4菱形区域
            for i in range(-1, 3):  # x偏移
                for j in range(-1, 3):  # y偏移
                    if abs(i) + abs(j) <= 2:  # 菱形区域条件
                        if self.is_obstacle(x + i, y + j):
                            return False
        else:
            # 水平或竖直移动，检查5×5区域
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if self.is_obstacle(x + i, y + j):
                        return False
        return True

    def heuristic(self, x1, y1, x2, y2):
        # 使用欧几里得距离作为启发式函数
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_neighbors(self, node):
        # 获取当前节点的所有可能邻居
        neighbors = []
        x, y = node.x, node.y

        # 8个可能的移动方向
        directions = [
            (0, 1, 1),  # 上
            (1, 0, 1),  # 右
            (0, -1, 1),  # 下
            (-1, 0, 1),  # 左
            (1, 1, math.sqrt(2)),  # 右上
            (1, -1, math.sqrt(2)),  # 右下
            (-1, 1, math.sqrt(2)),  # 左上
            (-1, -1, math.sqrt(2))  # 左下
        ]

        for dx, dy, cost in directions:
            nx, ny = x + dx, y + dy
            is_diagonal = (dx != 0 and dy != 0)

            # 检查新位置是否有效
            if self.is_valid_position(nx, ny, is_diagonal):
                neighbors.append((nx, ny, cost))

        return neighbors

    def find_path(self, start_x, start_y, goal_x, goal_y):
        # 将坐标转换为网格坐标
        start_x -= self.min_x
        start_y -= self.min_y
        goal_x -= self.min_x
        goal_y -= self.min_y

        # 检查起点和终点是否有效
        if not self.is_valid_position(start_x, start_y):
            raise ValueError("起点位置无效或位于障碍物上")

        if not self.is_valid_position(goal_x, goal_y):
            raise ValueError("终点位置无效或位于障碍物上")

        # 初始化开放列表和关闭列表
        open_list = []
        closed_set = set()

        # 创建起点和终点节点
        start_node = Node(start_x, start_y, 0, self.heuristic(start_x, start_y, goal_x, goal_y))
        goal_node = Node(goal_x, goal_y)

        # 将起点加入开放列表
        heapq.heappush(open_list, start_node)

        # 记录所有节点的字典，用于快速查找
        all_nodes = {(start_x, start_y): start_node}

        start_time = time.time()

        while open_list:
            # 获取f值最小的节点
            current_node = heapq.heappop(open_list)

            # 如果到达目标点，重构路径
            if current_node.x == goal_node.x and current_node.y == goal_node.y:
                path = []
                current = current_node
                while current is not None:
                    # 转换回原始坐标
                    path.append((current.x + self.min_x, current.y + self.min_y))
                    current = current.parent

                path.reverse()
                end_time = time.time()
                return path, current_node.g_cost, end_time - start_time

            # 将当前节点加入关闭列表
            closed_set.add((current_node.x, current_node.y))

            # 遍历所有邻居
            for nx, ny, move_cost in self.get_neighbors(current_node):
                if (nx, ny) in closed_set:
                    continue

                # 计算新的g成本
                tentative_g_cost = current_node.g_cost + move_cost

                # 检查是否已经存在该节点
                if (nx, ny) in all_nodes:
                    neighbor_node = all_nodes[(nx, ny)]
                    if tentative_g_cost < neighbor_node.g_cost:
                        # 找到更优路径，更新节点
                        neighbor_node.g_cost = tentative_g_cost
                        neighbor_node.parent = current_node
                        # 由于优先级改变，需要重新加入堆中
                        if neighbor_node in open_list:
                            open_list.remove(neighbor_node)
                            heapq.heapify(open_list)
                        heapq.heappush(open_list, neighbor_node)
                else:
                    # 创建新节点
                    h_cost = self.heuristic(nx, ny, goal_x, goal_y)
                    new_node = Node(nx, ny, tentative_g_cost, h_cost, current_node)
                    all_nodes[(nx, ny)] = new_node
                    heapq.heappush(open_list, new_node)

        # 开放列表为空，未找到路径
        end_time = time.time()
        return None, float('inf'), end_time - start_time

    def visualize_path(self, path):
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制网格
        for x in range(self.width):
            for y in range(self.height):
                color = self.grid[x][y]
                if color:
                    r, g, b = color
                    color_str = f'#{r:02x}{g:02x}{b:02x}'
                    rect = patches.Rectangle((x + self.min_x - 0.5, y + self.min_y - 0.5), 1, 1,
                                             linewidth=0.5, edgecolor='gray', facecolor=color_str, alpha=0.7)
                    ax.add_patch(rect)

        # 绘制路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'r-', linewidth=2, label='路径')
            ax.plot(path_x[0], path_y[0], 'bx', markersize=4, label='起点')
            ax.plot(path_x[-1], path_y[-1], 'bo', markersize=4, label='终点')

        # 设置图形属性
        ax.set_xlim(self.min_x - 1, self.max_x + 1)
        ax.set_ylim(self.min_y - 1, self.max_y + 1)
        ax.set_aspect('equal')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_title('车辆路径规划结果')
        ax.legend()

        plt.show()


# 主函数
def main():
    # 示例用法
    csv_file = "dapeng_1/rgb_grid_map_small.csv"  # 替换为您的CSV文件路径

    try:
        planner = AStarPathPlanner(csv_file)

        # 获取起点和终点坐标（示例值，需要根据实际情况修改）
        start_x, start_y = 19, 40
        goal_x, goal_y = 138, 40

        # 寻找路径
        path, length, time_used = planner.find_path(start_x, start_y, goal_x, goal_y)

        if path:
            print("路径规划成功！")
            print(f"路径点序列: {path}")
            print(f"路径总长度: {length:.4f}")
            print(f"规划用时: {time_used:.4f}秒")

            # 可视化路径
            planner.visualize_path(path)
        else:
            print("未找到可行路径！")

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()