import numpy as np
import time

# Define benchmark functions
def f1(position):
    return np.sum(position**2)

def f2(position):
    return np.sum(np.abs(position)) + np.prod(np.abs(position))

def f3(position):
    # Take the absolute value to ensure non-negative results
    abs_position = np.abs(position)
    n = len(abs_position)
    return np.sum([np.sum(abs_position[:i+1]) for i in range(n)])

def f4(position):
    return np.max(np.abs(position))

def f5(position):
    return np.sum(np.floor(position + 0.5)**2)

# Initialize algorithm parameters
num_agents = 100
max_iter = 3000
search_space = np.array([[-10, 10]] * 20)
num_empires = 10
num_runs = 30

# Main ICA function
def ica(num_agents, max_iter, search_space, num_empires, benchmark_function):
    num_dimensions = len(search_space)
    agents = np.random.uniform(search_space[:, 0], search_space[:, 1], (num_agents, num_dimensions))
    fitness = np.apply_along_axis(benchmark_function, 1, agents)

    # Select the empires based on fitness
    empire_indices = np.argsort(fitness)[:num_empires]
    empires = agents[empire_indices]
    colonies = np.delete(agents, empire_indices, axis=0)

    # Update fitness values for empires and colonies
    empire_fitness = np.apply_along_axis(benchmark_function, 1, empires)
    colony_fitness = np.apply_along_axis(benchmark_function, 1, colonies)

    # Initialize adaptive learning and assimilation rates
    learning_rate_init = 0.5
    learning_rate_final = 0.01
    assimilation_coeff_init = 0.5
    assimilation_coeff_final = 0.1

    for t in range(max_iter):

        # Update adaptive assimilation coefficient
        assimilation_coeff = assimilation_coeff_init - (assimilation_coeff_init - assimilation_coeff_final) * (t / max_iter)

        # Update adaptive learning rate
        learning_rate = learning_rate_init - (learning_rate_init - learning_rate_final) * (t / max_iter)

        # Perform crossover between empires
        for i in range(num_empires):
            if np.random.rand() < 0.5:
                other = np.random.randint(num_empires)
                child = 0.5 * (empires[i] + empires[other])
                child_fitness = benchmark_function(child)
                if child_fitness < empire_fitness[i]:
                    empires[i] = child
                    empire_fitness[i] = child_fitness

        # Update positions of colonies based on their respective empires
        for i, imperialist in enumerate(empires):
            colonies -= learning_rate * assimilation_coeff * (colonies - imperialist)

        # Perform revolution by introducing random perturbations
        colonies += np.random.normal(0, 0.2, colonies.shape)

        # Update fitness of all agents
        empire_fitness = np.apply_along_axis(benchmark_function, 1, empires)
        colony_fitness = np.apply_along_axis(benchmark_function, 1, colonies)

        # Find the best empire
        best_index = np.argmin(empire_fitness)
        best_agent = empires[best_index]
        best_fitness = empire_fitness[best_index]

        if np.min(empire_fitness) < 1e-9:
            break

    # Display progress for every 100th iteration
    if t % 100 == 0:
        print(f"Iteration {t}, Best Fitness: {best_fitness}")

    return best_agent, best_fitness

# Run the ICA algorithm and display the results
for benchmark_function in [f1, f2, f3, f4, f5]:
    fitness_results = []
    run_times = []
    for _ in range(num_runs):
        start_time = time.time()
        best_agent, fitness = ica(num_agents, max_iter, search_space, num_empires, benchmark_function)
        run_time = time.time() - start_time

        fitness_results.append(fitness)
        run_times.append(run_time)

    mean_fitness = np.mean(fitness_results)
    fitness_std = np.std(fitness_results)
    mean_run_time = np.mean(run_times)

    print(f"Benchmark function: {benchmark_function.__name__}")
    print(f"Average fitness: {mean_fitness}")
    print(f"Standard deviation: {fitness_std}")
    print(f"Average run time: {mean_run_time} sec")
    print()