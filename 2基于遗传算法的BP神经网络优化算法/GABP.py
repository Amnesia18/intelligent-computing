import numpy as np
import matplotlib.pyplot as plt
from math import pi
import random
import copy

# 超参数
popsize = 100  # 种群规模
Gmax = 500  # 最大迭代次数
pc = 0.8  # 交叉概率
pm = 0.1  # 变异概率
amax = 15  # 染色体基因值的上界
amin = -15  # 基因值下界

inputnum, outputnum = 2, 1  # 输入神经元，输出神经元个数
hiddennum = 10  # 隐层节点数

num = 1000  # 数据总数
trainnum = int(0.9 * num)  # 训练数据数量
testnum = int(0.1 * num)  # 测试数据数量

# 生成数据集
def generate_dataset(num):
    input_data = np.random.uniform(low=-2 * pi, high=2 * pi, size=[num, 2])
    output_data = 2 * np.multiply(input_data[:, 0], input_data[:, 0]) + np.sin(input_data[:, 1] + pi / 4)
    return input_data, output_data

# 数据归一化[0,1]
def normalize_data(input_data, output_data):
    output_max = np.max(output_data)
    output_min = np.min(output_data)
    input_datan = (input_data + 2 * pi) / (4 * pi)  # 将输入数据归一化到[0, 1]范围内
    output_datan = (output_data - output_min) / (output_max - output_min)
    return input_datan, output_datan, output_max, output_min

# 划分训练集和测试集

def split_data(input_datan, output_datan):
    input_train = input_datan[:trainnum, :]#训练集
    input_test = input_datan[trainnum:, :]#测试集
    out_train = output_datan[:trainnum]
    out_test = output_datan[trainnum:]
    return input_train, input_test, out_train, out_test

# 解码染色体为神经网络的权值和阈值
def decode(chrom):
    w_hide = chrom[:inputnum * hiddennum].reshape(inputnum, hiddennum)#输入层到隐层的权值
    bias_hide = chrom[inputnum * hiddennum: inputnum * hiddennum + hiddennum]#隐层的阈值
    w_out = chrom[inputnum * hiddennum + hiddennum: inputnum * hiddennum + hiddennum + hiddennum * outputnum].reshape(hiddennum, outputnum)#隐层到输出层的权值
    bias_out = chrom[inputnum * hiddennum + hiddennum + hiddennum * outputnum:]#输出层的阈值
    return w_hide, bias_hide, w_out, bias_out

# 个体类
class Individual:
    def __init__(self):
        self.L = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum#染色体长度
        self.chrom = np.random.uniform(low=amin, high=amax, size=self.L)#初始化染色体
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        w_hide, bias_hide, w_out, bias_out = decode(self.chrom)#将个体的染色体解码为神经网络的权重和阈值。
        hide_in = np.matmul(input_train, w_hide)#矩阵乘积
        hide_out = 1 / (1 + np.exp(-(hide_in - bias_hide)))#激活函数
        out_in = np.matmul(hide_out, w_out)#矩阵乘积
        y = 1 / (1 + np.exp(-(out_in - bias_out))).reshape(1, -1)#激活函数
        cost = np.abs(y - out_train)#损失函数，预测输出与实际输出之间的绝对误差
        sumcost = np.sum(cost)
        fitness = 1 / (1 + sumcost)  # 避免适应度为零
        return fitness

# 初始化种群
def init_population(popsize):
    return [Individual() for _ in range(popsize)]

# 寻找种群中的最优个体
def find_best(pop):
    return max(pop, key=lambda ind: ind.fitness)

# 选择操作
def select(pop):
    pop = sorted(pop, key=lambda ind: ind.fitness, reverse=True)#按适应度降序排序
    selected_pop = pop[:int(0.5 * len(pop))]#选择适应度最高的一半个体
    # 如果选择后的种群大小小于2，则补充随机个体
    while len(selected_pop) < 2:
        selected_pop.append(Individual())
    return selected_pop

# 交叉操作
def cross(pop, pc):
    new_pop = []
    while len(new_pop) < len(pop):
        if random.random() < pc:#以概率pc选择两个个体进行交叉
            parent1, parent2 = random.sample(pop, 2)
            point = random.randint(0, len(parent1.chrom) - 1)#随机选择交叉点
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            child1.chrom[:point], child2.chrom[:point] = parent2.chrom[:point], parent1.chrom[:point]
            new_pop.extend([child1, child2])
        else:
            new_pop.extend(random.sample(pop, 2))
    return new_pop[:len(pop)]

# 变异操作
def mutate(pop, pm, amax, amin, G, Gmax):
    for ind in pop:
        for i in range(len(ind.chrom)):
            if random.random() < pm:
                ind.chrom[i] += (amax - amin) * (random.random() - 0.5) * (1 - G / Gmax)#变异幅度范围，随机变异因子，衰减因子
    return pop

# BP算法,有效地调整神经网络的权重和偏置，使其能够更好地拟合训练数据。
def BP(input_train, out_train, w_hide, bias_hide, w_out, bias_out, hiddennum, trainnum):
    learning_rate = 0.01
    max_iter = 5000
    for _ in range(max_iter):
        # 前向传播
        hide_in = np.matmul(input_train, w_hide)
        hide_out = 1 / (1 + np.exp(-(hide_in - bias_hide)))#激活函数
        out_in = np.matmul(hide_out, w_out)
        y = 1 / (1 + np.exp(-(out_in - bias_out)))

        # 计算误差
        error = out_train - y.flatten()
        delta_out = error * y.flatten() * (1 - y.flatten())
        delta_hide = np.dot(delta_out.reshape(-1, 1), w_out.T) * hide_out * (1 - hide_out)

        # 更新权值和阈值
        w_out += learning_rate * np.dot(hide_out.T, delta_out.reshape(-1, 1))#更新隐层到输出层的权重
        bias_out += learning_rate * np.sum(delta_out)
        w_hide += learning_rate * np.dot(input_train.T, delta_hide)#更新输入层到隐层的权重
        bias_hide += learning_rate * np.sum(delta_hide, axis=0)
    return w_hide, bias_hide, w_out, bias_out

# 画标准图像
def plot_standardimage():
    x = np.linspace(-2 * pi, 2 * pi, 100)
    y = np.linspace(-2 * pi, 2 * pi, 100)
    X, Y = np.meshgrid(x, y)
    Z = 2 * X * X + np.sin(Y + pi / 4)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.title("Standard Function")
    plt.show()

# 画拟合图像
def plot_fittingimage(w_hide, bias_hide, w_out, bias_out, output_max, output_min, title):
    x = np.linspace(-2 * pi, 2 * pi, 100)
    y = np.linspace(-2 * pi, 2 * pi, 100)
    X, Y = np.meshgrid(x, y)
    input_data = np.stack((X.flatten(), Y.flatten()), axis=1)
    hide_in = np.matmul(input_data, w_hide)
    hide_out = 1 / (1 + np.exp(-(hide_in - bias_hide)))
    out_in = np.matmul(hide_out, w_out)
    Z = 1 / (1 + np.exp(-(out_in - bias_out)))
    Z = (output_max - output_min) * Z + output_min
    Z = Z.reshape(X.shape)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.title(title)
    plt.show()

# 画出神经网络的预测结果
def plot_results(w_hide, bias_hide, w_out, bias_out, title, output_max, output_min, input_test, out_test):
    output = []
    for m in range(testnum):
        hide_in = np.matmul(input_test[m], w_hide)
        hide_out = 1 / (1 + np.exp(-(hide_in - bias_hide)))
        out_in = np.matmul(hide_out, w_out)
        c = 1 / (1 + np.exp(-(out_in - bias_out)))#
        output.append(c[0])
    output_un = (output_max - output_min) * np.array(output) + output_min
    MAD = np.sum(abs(output_un - out_test)) / testnum
    print(f"\n{title}算法测试的平均绝对误差为：", MAD)
    if title == "GA":
        plot_standardimage()
    plt.ion()
    ax = plt.axes(projection='3d')
    ax.scatter3D(2 * pi * input_test[:, 0], 2 * pi * input_test[:, 1], output_un, 'binary')
    plt.title(f"The test result of {title}\nTurn off automatically after 5 seconds")
    plt.pause(5)
    plt.close()

# 主程序
input_data, output_data = generate_dataset(num)
input_datan, output_datan, output_max, output_min = normalize_data(input_data, output_data)
input_train, input_test, out_train, out_test = split_data(input_datan, output_datan)
pop = init_population(popsize)
ind_best_global = find_best(pop)
best_fit_iteration = [ind_best_global.fitness]

for G in range(1, Gmax + 1):
    # print(f"--------------第{G}次迭代--------------")
    pop = select(pop)
    pop = cross(pop, pc)
    mutate(pop, pm, amax, amin, G, Gmax)
    ind_best_now = find_best(pop)
    if ind_best_now.fitness > ind_best_global.fitness:
        ind_best_global = copy.deepcopy(ind_best_now)
    # print(f"当前最优适应度：{ind_best_now.fitness}")
    # print(f"全局最优适应度：{ind_best_global.fitness}")
    best_fit_iteration.append(ind_best_global.fitness)

w_hide, bias_hide, w_out, bias_out = decode(ind_best_global.chrom)
plot_results(w_hide, bias_hide, w_out, bias_out, "GA", output_max, output_min, input_test, out_test)


w_hide_bp, bias_hide_bp, w_out_bp, bias_out_bp = BP(input_train, out_train, w_hide, bias_hide, w_out, bias_out, hiddennum, trainnum)
plot_results(w_hide_bp, bias_hide_bp, w_out_bp, bias_out_bp, "GABP", output_max, output_min, input_test, out_test)
