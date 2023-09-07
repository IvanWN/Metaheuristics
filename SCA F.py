import numpy as np
import time

def objective_function(position):
    # Здесь необходимо определить целевую функцию для оптимизации
    return np.sum(position**2)

def sca(num_agents, max_iter, search_space, a_linear_component):
    num_dimensions = len(search_space)

    agents = np.random.uniform(search_space[:, 0], search_space[:, 1], (num_agents, num_dimensions))

    fitness = np.apply_along_axis(objective_function, 1, agents)
    best_agent_index = np.argmin(fitness)
    best_agent_fitness = fitness[best_agent_index]
    best_agent = agents[best_agent_index]

    for t in range(max_iter):
        for i in range(num_agents):
            a_t = a_linear_component - t * (a_linear_component / max_iter)

            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            r4 = np.random.random()
            A = 2 * a_t * r1 - a_t
            C = 2 * r2

            random_agent_index = np.random.randint(num_agents)
            while random_agent_index == i:
                random_agent_index = np.random.randint(num_agents)

            random_agent = agents[random_agent_index]

            D = np.abs(C * random_agent - agents[i])
            new_position = random_agent - A * D

            if objective_function(new_position) < fitness[i]:
                agents[i] = new_position
                fitness[i] = objective_function(new_position)

                if fitness[i] < best_agent_fitness:
                    best_agent_fitness = fitness[i]
                    best_agent = agents[i]

    return best_agent, best_agent_fitness

def sphere(position):
    return np.sum(position**2)

def schwefels_problem_2_22(position):
    return np.sum(np.abs(position)) + np.prod(np.abs(position))

def schwefels_problem_1_2(position):
    n = len(position)
    return np.sum([np.sum(position[:i+1]) for i in range(n)])

def schwefels_problem_2_21(position):
    return np.max(np.abs(position))

def rosenbrock(position):
    return np.sum(100 * (position[1:] - position[:-1]**2)**2 + (position[:-1] - 1)**2)

def step_function(position):
    return np.sum(np.floor(position + 0.5)**2)

def quartic_function(position):
    d = len(position)
    return np.sum(np.arange(1, d+1) * (position**4))

def schwefels_problem_2_26(position):
    return np.sum(-position * np.sin(np.sqrt(np.abs(position))))

def rastrigin_function(position):
    return np.sum(position**2 - 10 * np.cos(2 * np.pi * position) + 10)

def ackley_function(position):
    d = len(position)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(position**2) / d)) - np.exp(np.sum(np.cos(2 * np.pi * position)) / d) + 20 + np.exp(1)

def griewank_function(position):
    d = len(position)
    return np.sum(position**2) / 4000 - np.prod(np.cos(position / np.sqrt(np.arange(1, d+1)))) + 1

def levy_function(position):
    d = len(position)
    y = 1 + (position - 1) / 4

    term1 = np.sin(np.pi * y[0])**2
    term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
    term3 = (y[-1] - 1)**2 * (1 + np.sin(2 * np.pi * y[-1])**2)

    return np.pi / d * (10 * term1 + term2 + term3)

def u(x, a, k, m):
    return k * ((x - a)**m * (x > a) + (-k) * (x + a)**m * (x < -a))

def penalized_function(position):
    d = len(position)
    y = 1 + (position - 1) / 4

    term1 = np.sin(3 * np.pi * y[0])
    term2 = np.sum((y[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * y[1:])**2))
    term3 = (y[-1] - 1)**2 * (1 + np.sin(2 * np.pi * y[-1])**2)

    return 0.1 * (term1 + term2 + term3) + np.sum(u(position, 5, 100, 4))

def weierstrass_function(position):
    a = 0.5
    b = 3
    K = 20
    k_arr = np.arange(0, K+1)
    x_arr = np.expand_dims(position, axis=0)

    sum_k = np.sum(a**k_arr * np.cos(2 * np.pi * b**k_arr * (x_arr + 0.5)), axis=1)
    sum_i = np.sum(sum_k)

    # Compute the bias term
    d = len(position)
    bias = d * np.sum(a**k_arr * np.cos(2 * np.pi * b**k_arr * 0.5))

    return sum_i - bias

# Определим функцию для расчета сигмы
def sigma(y, ro, lamda):
    return ro * ((y - lamda) ** 2)

# Определим функцию для расчета omega
def omega(y, fi, ro, lamda):
    return np.sum([fi[i] * sigma(y[i], ro[i], lamda[i]) for i in range(len(fi))])

# Определим наши композитные функции
def CF1(x):
    fi = [sphere] * 10
    lamda = [5/100] * 10
    ro = [1] * 10
    y = x  # Для CF1 y = x
    return omega(y, fi, ro, lamda)

def CF2(x):
    fi = [griewank_function] * 10
    lamda = [5/100] * 10
    ro = [1] * 10
    y = x  # Для CF2 y = x
    return omega(y, fi, ro, lamda)

def CF3(x):
    fi = [griewank_function] * 10
    lamda = [1] * 10
    ro = [1] * 10
    y = x  # Для CF3 y = x
    return omega(y, fi, ro, lamda)

def CF4(x):
    fi = [ackley_function] * 2 + [rastrigin_function] * 2 + [griewank_function] * 2 + [sphere] * 4
    lamda = [5/32, 5/32, 1, 1, 5/0.5, 5/0.5, 5/100, 5/100, 5/100, 5/100]
    ro = [1] * 10
    y = x  # Для CF4 y = x
    return omega(y, fi, ro, lamda)

def CF5(x):
    fi = [rastrigin_function]*2 + [weierstrass_function]*2 + [griewank_function]*2 + [ackley_function]*2 + [sphere]*2
    lamda = [1/5]*2 + [5/0.5]*2 + [5/100]*2 + [5/32]*2 + [5/100]*2
    ro = [1] * 10
    y = x  # Для CF5 y = x
    return omega(y, fi, ro, lamda)

def CF6(x):
    fi = [rastrigin_function]*2 + [weierstrass_function]*2 + [griewank_function]*2 + [ackley_function]*2 + [sphere]*2
    lamda = [0.1 * 1/5, 0.2 * 1/5, 0.3 * 5/0.5, 0.4 * 5/0.5, 0.5 * 5/100, 0.6 * 5/100, 0.7 * 5/32, 0.8 * 5/32, 0.9 * 5/100, 1 * 5/100]
    ro = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    y = x  # Для CF6 y = x
    return omega(y, fi, ro, lamda)

benchmark_functions = [sphere, schwefels_problem_2_22, schwefels_problem_1_2, schwefels_problem_2_21, rosenbrock,
                       step_function, quartic_function, schwefels_problem_2_26, rastrigin_function, ackley_function,
                       griewank_function, levy_function, penalized_function, weierstrass_function, 
                       CF1, CF2, CF3, CF4, CF5, CF6]

def run_sca_benchmark(sca_function, benchmark_function, num_agents, max_iter, search_space, a_linear_component):
    def objective_function(position):
        return benchmark_function(position)

    sca_function.obj_func = objective_function
    best_agent, best_agent_fitness = sca_function(num_agents, max_iter, search_space, a_linear_component)
    return best_agent_fitness

num_agents = 20
max_iter = 1000
search_space = np.array([[-100, 100]] * 20)
a_linear_component = 2
num_runs = 30

for benchmark_function in benchmark_functions:
    fitness_results = []
    run_times = []

    for _ in range(num_runs):
        start_time = time.time()
        fitness = run_sca_benchmark(sca, benchmark_function, num_agents, max_iter, search_space, a_linear_component)
        run_time = time.time() - start_time

        fitness_results.append(fitness)
        run_times.append(run_time)

    mean_fitness = np.mean(fitness_results)
    fitness_std = np.std(fitness_results)
    mean_run_time = np.mean(run_times)

    print(f"Функция бенчмарка: {benchmark_function.__name__}")
    print(f"Среднее значение: {mean_fitness}")
    print(f"Ошибка (стандартное отклонение): {fitness_std}")
    print(f"Среднее время работы: {mean_run_time} сек")
    print()