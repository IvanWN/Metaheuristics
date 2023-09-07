import numpy as np
import time

def objective_function(position):
    # Define the objective function for optimization.
    return np.sum(position ** 2)

def sca(num_agents, max_iter, search_space, a_linear_component):
    # Initialize dimensions and agent positions in search space.
    num_dimensions = len(search_space)
    agents = np.random.uniform(search_space[:, 0], search_space[:, 1], (num_agents, num_dimensions))

    # Evaluate fitness of all agents.
    fitness = np.apply_along_axis(objective_function, 1, agents)
    best_agent_index = np.argmin(fitness)
    best_agent_fitness = fitness[best_agent_index]
    best_agent = agents[best_agent_index]

    # Optimization loop.
    for t in range(max_iter):
        for i in range(num_agents):
            a_t = a_linear_component - t * (a_linear_component / max_iter)

            # Generate random values for equations.
            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            r4 = np.random.random()
            A = 2 * a_t * r1 - a_t
            C = 2 * r2

            # Select a random agent different from current.
            random_agent_index = np.random.randint(num_agents)
            while random_agent_index == i:
                random_agent_index = np.random.randint(num_agents)
            random_agent = agents[random_agent_index]

            # Compute new position.
            D = np.abs(C * random_agent - agents[i])
            new_position = random_agent - A * D

            # Update agent's position if its fitness is improved.
            if objective_function(new_position) < fitness[i]:
                agents[i] = new_position
                fitness[i] = objective_function(new_position)

                # Update best agent if current agent's fitness is better.
                if fitness[i] < best_agent_fitness:
                    best_agent_fitness = fitness[i]
                    best_agent = agents[i]

    return best_agent, best_agent_fitness

# Below are various benchmark functions for optimization.

def sphere(position):
    return np.sum(position ** 2)

def schwefels_problem_2_22(position):
    return np.sum(np.abs(position)) + np.prod(np.abs(position))

def schwefels_problem_1_2(position):
    n = len(position)
    return np.sum([np.sum(position[:i + 1]) for i in range(n)])

def schwefels_problem_2_21(position):
    return np.max(np.abs(position))

def rosenbrock(position):
    return np.sum(100 * (position[1:] - position[:-1] ** 2) ** 2 + (position[:-1] - 1) ** 2)

def step_function(position):
    return np.sum(np.floor(position + 0.5) ** 2)

def quartic_function(position):
    d = len(position)
    return np.sum(np.arange(1, d + 1) * (position ** 4))

def schwefels_problem_2_26(position):
    return np.sum(-position * np.sin(np.sqrt(np.abs(position))))

def rastrigin_function(position):
    return np.sum(position ** 2 - 10 * np.cos(2 * np.pi * position) + 10)

def ackley_function(position):
    d = len(position)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(position ** 2) / d)) - np.exp(
        np.sum(np.cos(2 * np.pi * position)) / d) + 20 + np.exp(1)

def griewank_function(position):
    d = len(position)
    return np.sum(position ** 2) / 4000 - np.prod(np.cos(position / np.sqrt(np.arange(1, d + 1)))) + 1

def levy_function(position):
    d = len(position)
    y = 1 + (position - 1) / 4

    # Compute the three terms for the Levy function.
    term1 = np.sin(np.pi * y[0])**2
    term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
    term3 = (y[-1] - 1)**2 * (1 + np.sin(2 * np.pi * y[-1])**2)

    return np.pi / d * (10 * term1 + term2 + term3)

def penalized_function(position):
    d = len(position)
    y = 1 + (position - 1) / 4

    # Compute the three terms for the penalized function.
    term1 = np.sin(3 * np.pi * y[0])
    term2 = np.sum((y[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * y[1:])**2))
    term3 = (y[-1] - 1)**2 * (1 + np.sin(2 * np.pi * y[-1])**2)

    return 0.1 * (term1 + term2 + term3) + np.sum(u(position, 5, 100, 4))

def weierstrass_function(position):
    a, b, K = 0.5, 3, 20
    k_arr = np.arange(0, K+1)
    x_arr = np.expand_dims(position, axis=0)

    # Compute Weierstrass function value.
    sum_k = np.sum(a**k_arr * np.cos(2 * np.pi * b**k_arr * (x_arr + 0.5)), axis=1)
    sum_i = np.sum(sum_k)

    # Bias term for normalization.
    d = len(position)
    bias = d * np.sum(a**k_arr * np.cos(2 * np.pi * b**k_arr * 0.5))

    return sum_i - bias

# Define sigma for the composite functions.
def sigma(y, ro, lamda):
    return ro * ((y - lamda) ** 2)

# Define omega (weight function) for the composite functions.
def omega(y, fi, ro, lamda):
    return np.sum([fi[i] * sigma(y[i], ro[i], lamda[i]) for i in range(len(fi))])

# Composite functions (CF) built using other benchmark functions.
def CF1(x):
    # For CF1, using sphere function.
    fi = [sphere] * 10
    lamda, ro = [5/100] * 10, [1] * 10
    y = x
    return omega(y, fi, ro, lamda)

def CF2(x):
    # For CF2, using griewank function.
    fi = [griewank_function] * 10
    lamda, ro = [5/100] * 10, [1] * 10
    y = x
    return omega(y, fi, ro, lamda)

def CF3(x):
    # For CF3, using griewank function.
    fi = [griewank_function] * 10
    lamda, ro = [1] * 10, [1] * 10
    y = x
    return omega(y, fi, ro, lamda)

def CF4(x):
    # For CF4, using a mix of benchmark functions.
    fi = [ackley_function] * 2 + [rastrigin_function] * 2 + [griewank_function] * 2 + [sphere] * 4
    lamda = [5/32, 5/32, 1, 1, 5/0.5, 5/0.5, 5/100, 5/100, 5/100, 5/100]
    ro = [1] * 10
    y = x
    return omega(y, fi, ro, lamda)

def CF5(x):
    # For CF5, using a mix of benchmark functions.
    fi = [rastrigin_function]*2 + [weierstrass_function]*2 + [griewank_function]*2 + [ackley_function]*2 + [sphere]*2
    lamda, ro = [1/5]*2 + [5/0.5]*2 + [5/100]*2 + [5/32]*2 + [5/100]*2, [1] * 10
    y = x
    return omega(y, fi, ro, lamda)

def CF6(x):
    # Define composite benchmark function CF6.
    fi = [rastrigin_function]*2 + [weierstrass_function]*2 + [griewank_function]*2 + [ackley_function]*2 + [sphere]*2
    lamda = [0.1 * 1/5, 0.2 * 1/5, 0.3 * 5/0.5, 0.4 * 5/0.5, 0.5 * 5/100, 0.6 * 5/100, 0.7 * 5/32, 0.8 * 5/32, 0.9 * 5/100, 1 * 5/100]
    ro = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    y = x  # For CF6, y is equal to x
    return omega(y, fi, ro, lamda)

# List of benchmark functions.
benchmark_functions = [sphere, schwefels_problem_2_22, schwefels_problem_1_2, schwefels_problem_2_21, rosenbrock,
                       step_function, quartic_function, schwefels_problem_2_26, rastrigin_function, ackley_function,
                       griewank_function, levy_function, penalized_function, weierstrass_function,
                       CF1, CF2, CF3, CF4, CF5, CF6]

def run_sca_benchmark(sca_function, benchmark_function, num_agents, max_iter, search_space, a_linear_component):
    # Objective function wraps the benchmark function.
    def objective_function(position):
        return benchmark_function(position)

    sca_function.obj_func = objective_function
    best_agent, best_agent_fitness = sca_function(num_agents, max_iter, search_space, a_linear_component)
    return best_agent_fitness

num_agents = 20
max_iter = 1000
# Define search space as 20 dimensions, each ranging from -100 to 100.
search_space = np.array([[-100, 100]] * 20)
a_linear_component = 2
num_runs = 30

# Loop over all benchmark functions.
for benchmark_function in benchmark_functions:
    fitness_results = []
    run_times = []

    # Run the optimization multiple times for each benchmark.
    for _ in range(num_runs):
        start_time = time.time()
        fitness = run_sca_benchmark(sca, benchmark_function, num_agents, max_iter, search_space, a_linear_component)
        run_time = time.time() - start_time

        fitness_results.append(fitness)
        run_times.append(run_time)

    # Calculate and display mean and standard deviation for each benchmark.
    mean_fitness = np.mean(fitness_results)
    fitness_std = np.std(fitness_results)
    mean_run_time = np.mean(run_times)

    print(f"Benchmark function: {benchmark_function.__name__}")
    print(f"Mean fitness value: {mean_fitness}")
    print(f"Standard deviation: {fitness_std}")
    print(f"Average run time: {mean_run_time} seconds")
    print()
