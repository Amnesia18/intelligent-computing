import random
import numpy as np
import matplotlib.pyplot as plt

# 定义城市和计算距离的方法
class City:
    def __init__(self, x, y):
        self.x = x  # 城市的x坐标
        self.y = y  # 城市的y坐标

    def distance(self, city):
        # 计算当前城市与另一个城市之间的欧几里得距离
        return np.sqrt((self.x - city.x) ** 2 + (self.y - city.y) ** 2)

# 创建城市之间的距离矩阵
def create_distance_matrix(cities):
    size = len(cities)
    distance_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            distance_matrix[i][j] = cities[i].distance(cities[j])
    return distance_matrix

# 定义遗传算法的主要步骤
class GeneticAlgorithm:
    def __init__(self, cities, population_size=100, mutation_rate=0.01, generations=500):
        self.cities = cities  # 城市列表
        self.distance_matrix = create_distance_matrix(cities)  # 城市距离矩阵
        self.population_size = population_size  # 种群规模
        self.mutation_rate = mutation_rate  # 变异率
        self.generations = generations  # 迭代次数

    # 初始化种群，生成随机路径
    def initial_population(self):
        population = []
        for _ in range(self.population_size):
            individual = list(np.random.permutation(len(self.cities)))#通过对城市索引进行随机排列，创建多样化的初始种群
            population.append(individual)
        return population

    # 计算个体的适应度，适应度为路径长度的倒数
    def fitness(self, individual):
        return 1 / self.route_distance(individual)

    # 计算路径的总距离
    def route_distance(self, individual):
        distance = 0
        for i in range(len(individual)):
            from_city = individual[i]
            to_city = individual[(i + 1) % len(individual)]
            distance += self.distance_matrix[from_city][to_city]
        return distance

    # 选择适应度高的个体进行繁殖
    def selection(self, population, fitnesses):
        selected = random.choices(population, weights=fitnesses, k=self.population_size)
        return selected

    # 部分匹配交叉 (PMX) 操作生成子代
    def crossover(self, parent1, parent2):
        start = random.randint(0, len(parent1) - 1)
        end = random.randint(start, len(parent1) - 1)
        child = [-1] * len(parent1)
        child[start:end + 1] = parent1[start:end + 1]# 区间内的基因复制到子代的相应位置

        for gene in parent2:
            if gene not in child:
                for i in range(len(child)):
                    if child[i] == -1:
                        child[i] = gene
                        break
        return child

    # 变异操作，以一定概率交换个体的两个城市
    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:#生成一个随机数，如果该随机数小于变异率，则进行变异操作。
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    # 遗传算法主循环
    def evolve(self):
        population = self.initial_population()
        initial_route = population[0]
        for _ in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]
            selected = self.selection(population, fitnesses)
            next_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]#通过取模操作确保索引不越界
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                next_population.append(self.mutate(child1))
                next_population.append(self.mutate(child2))
            population = next_population
        best_individual = max(population, key=self.fitness)
        return initial_route, best_individual, self.route_distance(best_individual)

# 绘制路径
def plot_route(cities, route, title):
    x = [cities[i].x for i in route]
    y = [cities[i].y for i in route]
    plt.plot(x, y, 'o-', color='red')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# 示例运行
def main():
    # 随机生成20个城市
    cities = [City(x=round(random.uniform(0, 100)), y=round(random.uniform(0, 100))) for _ in range(20)]
    # 初始化遗传算法
    ga = GeneticAlgorithm(cities)
    # 运行遗传算法
    initial_route, best_route, best_distance = ga.evolve()
    # 输出初始路径和最优路径，以及最优距离
    print("Initial route:", initial_route)
    print("Best route:", best_route)
    print("Best distance:", best_distance)
    # 绘制初始路径
    plot_route(cities, initial_route, "Initial Route")
    # 绘制最佳路径
    plot_route(cities, best_route, f"Best Route (Distance: {best_distance})")

if __name__ == "__main__":
    main()
