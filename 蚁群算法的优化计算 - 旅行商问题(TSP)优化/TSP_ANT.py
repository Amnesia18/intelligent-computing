import numpy as np
import matplotlib.pyplot as plt

# 生成20个城市的坐标，范围调整为[0,100]
num_cities = 20
cities = np.random.rand(num_cities, 2) * 100

# 计算城市间距离矩阵
def calc_distance_matrix(cities):
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist_matrix
dist_matrix = calc_distance_matrix(cities)

# 蚁群算法参数
num_ants = 50#蚂蚁数量
num_iterations = 100 # 迭代次数
alpha = 1.0   # 信息素的重要性
beta = 2.0    # 启发因子的相对重要性
evaporation_rate = 0.5#信息素蒸发率
Q = 100.0# 信息素增加强度

# 初始化信息素矩阵
pheromone = np.ones((num_cities, num_cities))

# 计算启发因子矩阵
visibility = 1.0 / (dist_matrix + np.eye(num_cities))

# 运行蚁群算法
best_length = float('inf')
best_path = []

for iteration in range(num_iterations):
    paths = []
    lengths = []
#     每只蚂蚁都会走一遍所有城市
    for ant in range(num_ants):
        visited = np.zeros(num_cities, dtype=bool)
        path = []
        current_city = np.random.randint(0, num_cities)
        path.append(current_city)
        visited[current_city] = True
#         选择下一个城市并构建完整路径
        for _ in range(num_cities - 1):
            probabilities = []
            for next_city in range(num_cities):
                if not visited[next_city]:
                    probabilities.append(
                        (pheromone[current_city][next_city] ** alpha) *
                        (visibility[current_city][next_city] ** beta)
                    )
                else:
                    probabilities.append(0)
            probabilities = np.array(probabilities) / np.sum(probabilities)
            next_city = np.random.choice(range(num_cities), p=probabilities)
            path.append(next_city)
            visited[next_city] = True
            current_city = next_city

        path.append(path[0])  # 回到起点
        paths.append(path)
        lengths.append(sum(dist_matrix[path[i]][path[i + 1]] for i in range(num_cities)))
#. 更新全局最优路径
    for i, path in enumerate(paths):
        length = lengths[i]
        if length < best_length:
            best_length = length
            best_path = path
#更新信息素
    pheromone *= (1 - evaporation_rate)
    for i, path in enumerate(paths):
        length = lengths[i]
        for j in range(num_cities):
            pheromone[path[j]][path[j + 1]] += Q / length

# 显示最佳路径
plt.figure()
plt.scatter(cities[:, 0], cities[:, 1])
for i in range(num_cities):
    plt.annotate(i, (cities[i, 0], cities[i, 1]))
best_path_cities = cities[best_path]
plt.plot(best_path_cities[:, 0], best_path_cities[:, 1], 'r-')
plt.title(f'Best path length: {best_length}')
plt.show()

# 输出最优距离和路径
print(f'最优距离: {best_length}')
print('最优路径:')
print(' -> '.join(map(str, best_path)))
