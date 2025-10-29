import csv
import math
import time
import heapq
import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self, x, y, g_cost=float('inf'), h_cost=0, parent=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost  # 从起点到当前节点的实际代价
        self.h_cost = h_cost  # 从当前节点到终点的启发式代价
        self.parent = parent  # 父节点，用于回溯路径

    def f_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost() < other.f_cost()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class AStarPlanner:
    def __init__(self, grid, vehicle_size=5):
        self.grid = grid
        self.vehicle_size = vehicle_size
        self.half_vehicle = vehicle_size // 2
        self.width = len(grid)
        self.height = len(grid[0])

    def is_valid_position(self, x, y):
        """检查车辆在(x,y)位置时是否会碰到障碍物"""
        # 计算车辆覆盖的区域
        x_start = max(0, x - self.half_vehicle)
        x_end = min(self.width, x + self.half_vehicle + 1)
        y_start = max(0, y - self.half_vehicle)
        y_end = min(self.height, y + self.half_vehicle + 1)

        # 检查区域内是否有障碍物
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                if self.grid[i][j] == 1:  # 1表示障碍物
                    return False
        return True

    def heuristic(self, x1, y1, x2, y2):
        """计算启发式代价（使用欧几里得距离）"""
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_neighbors(self, node):
        """获取当前节点的所有可行邻居节点"""
        neighbors = []
        # 8个可能的移动方向：上、下、左、右、左上、右上、左下、右下
        directions = [
            (0, 1, 1.0),  # 上
            (1, 0, 1.0),  # 右
            (0, -1, 1.0),  # 下
            (-1, 0, 1.0),  # 左
            (1, 1, math.sqrt(2)),  # 右上
            (1, -1, math.sqrt(2)),  # 右下
            (-1, 1, math.sqrt(2)),  # 左上
            (-1, -1, math.sqrt(2))  # 左下
        ]

        for dx, dy, cost in directions:
            nx, ny = node.x + dx, node.y + dy

            # 检查新位置是否在地图范围内且没有障碍物
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.is_valid_position(nx, ny):
                    neighbors.append((nx, ny, cost))

        return neighbors

    def plan(self, start_x, start_y, goal_x, goal_y):
        """执行A*算法规划路径"""
        start_time = time.time()

        # 创建起点和终点节点
        start_node = Node(start_x, start_y, 0, self.heuristic(start_x, start_y, goal_x, goal_y))
        goal_node = Node(goal_x, goal_y)

        # 初始化开放列表和关闭列表
        open_list = []
        closed_set = set()

        # 将起点加入开放列表
        heapq.heappush(open_list, start_node)

        # 记录所有节点的字典，用于快速查找和更新
        all_nodes = {(start_x, start_y): start_node}

        while open_list:
            # 获取f值最小的节点
            current_node = heapq.heappop(open_list)

            # 如果到达目标点，回溯路径
            if current_node.x == goal_node.x and current_node.y == goal_node.y:
                path = []
                total_length = 0
                current = current_node

                while current:
                    path.append((current.x, current.y))
                    if current.parent:
                        # 计算到父节点的距离
                        dx = current.x - current.parent.x
                        dy = current.y - current.parent.y
                        if abs(dx) == 1 and abs(dy) == 1:
                            total_length += math.sqrt(2)
                        else:
                            total_length += 1.0
                    current = current.parent

                path.reverse()
                planning_time = time.time() - start_time
                return path, total_length, planning_time

            # 将当前节点加入关闭列表
            closed_set.add((current_node.x, current_node.y))

            # 遍历所有邻居
            for nx, ny, move_cost in self.get_neighbors(current_node):
                if (nx, ny) in closed_set:
                    continue

                # 计算新的g值
                tentative_g_cost = current_node.g_cost + move_cost

                # 检查邻居节点是否已经在开放列表中
                neighbor_node = all_nodes.get((nx, ny), None)

                if neighbor_node is None:
                    # 新节点，创建并加入开放列表
                    h_cost = self.heuristic(nx, ny, goal_x, goal_y)
                    neighbor_node = Node(nx, ny, tentative_g_cost, h_cost, current_node)
                    heapq.heappush(open_list, neighbor_node)
                    all_nodes[(nx, ny)] = neighbor_node
                elif tentative_g_cost < neighbor_node.g_cost:
                    # 找到更优路径，更新节点
                    neighbor_node.g_cost = tentative_g_cost
                    neighbor_node.parent = current_node
                    # 由于节点的f值可能改变，需要重新排序开放列表
                    # 这里采用简单方法：将节点再次加入堆中（会有重复，但通过关闭列表避免重复处理）
                    heapq.heappush(open_list, neighbor_node)

        # 如果没有找到路径
        planning_time = time.time() - start_time
        return None, 0, planning_time


def read_map_from_csv(filename):
    """从CSV文件读取地图数据"""
    grid = []
    color_map = {
        (0, 0, 0): 1,  # 黑色 - 障碍物
        (127, 0, 0): 1,  # 红色 - 障碍物
        (0, 127, 0): 0,  # 绿色 - 可通行
        (127, 127, 0): 0  # 黄色 - 可通行
    }

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x = int(row[0])
            y = int(row[1])
            r = int(row[2])
            g = int(row[3])
            b = int(row[4])

            # 扩展网格大小以容纳所有点
            while len(grid) <= x:
                grid.append([])
            while len(grid[x]) <= y:
                grid[x].append(None)

            # 根据颜色设置网格值
            grid[x][y] = color_map.get((r, g, b), 1)  # 默认视为障碍物

    # 确保网格是矩形，填充缺失的值为障碍物
    max_x = len(grid) - 1
    max_y = 0
    for x in range(len(grid)):
        if len(grid[x]) > max_y:
            max_y = len(grid[x]) - 1

    for x in range(len(grid)):
        for y in range(len(grid[x]), max_y + 1):
            grid[x].append(1)  # 填充为障碍物

    return grid


def visualize_path(grid, path, start, goal):
    """可视化地图和路径"""
    plt.figure(figsize=(10, 10))

    # 创建地图可视化数组
    vis_grid = np.zeros((len(grid), len(grid[0])))
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            vis_grid[x, y] = grid[x][y]

    # 绘制地图
    plt.imshow(vis_grid.T, cmap='binary', origin='lower')

    # 绘制路径
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, 'r-', linewidth=2)
        plt.plot(path_x, path_y, 'ro', markersize=3)

    # 标记起点和终点
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'bo', markersize=10, label='Goal')

    plt.legend()
    plt.title('Path Planning Result')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()


def main():
    # 读取地图数据
    filename = 'dapeng_1/rgb_grid_map_small_1.csv'  # 替换为您的CSV文件路径
    grid = read_map_from_csv(filename)

    # 获取起点和终点
    # 获取起点和终点坐标（示例值，需要根据实际情况修改）
    start_x, start_y = 19, 40
    goal_x, goal_y = 138, 40

    # 创建路径规划器
    planner = AStarPlanner(grid, vehicle_size=9)

    # 执行路径规划
    path, length, planning_time = planner.plan(start_x, start_y, goal_x, goal_y)

    # 输出结果
    if path:
        print("\n路径规划成功!")
        print(f"路径总长度: {length:.4f}")
        print(f"规划用时: {planning_time:.4f}秒")
        print("车辆行驶的中心点轨迹:")
        for i, (x, y) in enumerate(path):
            print(f"步骤{i}: ({x}, {y})")

        # 可视化结果
        visualize_path(grid, path, (start_x, start_y), (goal_x, goal_y))
    else:
        print("\n无法找到可行路径!")


if __name__ == "__main__":
    main()