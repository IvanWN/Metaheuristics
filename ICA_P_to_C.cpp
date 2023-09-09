#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <ctime>
#include <chrono>
#include <functional>

// Define benchmark functions
double f1(const std::vector<double>& position) {
    double sum = 0.0;
    for (double val : position) {
        sum += val * val;
    }
    return sum;
}

double f2(const std::vector<double>& position) {
    double sum = std::accumulate(position.begin(), position.end(), 0.0, [](double acc, double val) {
        return acc + std::abs(val);
    });
    double prod = std::accumulate(position.begin(), position.end(), 1.0, [](double acc, double val) {
        return acc * std::abs(val);
    });
    return sum + prod;
}

double f3(const std::vector<double>& position) {
    double result = 0.0;
    double sum = 0.0;
    for (double val : position) {
        sum += std::abs(val);
        result += sum;
    }
    return result;
}

double f4(const std::vector<double>& position) {
    double maxVal = 0.0;
    for (double val : position) {
        if (std::abs(val) > maxVal) {
            maxVal = std::abs(val);
        }
    }
    return maxVal;
}

double f5(const std::vector<double>& position) {
    double sum = 0.0;
    for (double val : position) {
        sum += std::pow(std::floor(val + 0.5), 2);
    }
    return sum;
}

// Main ICA function
std::pair<std::vector<double>, double> ica(int num_agents, int max_iter, const std::vector<std::pair<double, double>>& search_space,
                                           int num_empires, double (*benchmark_function)(const std::vector<double>&)) {
    int num_dimensions = search_space.size();
    std::vector<std::vector<double>> agents(num_agents, std::vector<double>(num_dimensions));
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    for (int i = 0; i < num_agents; ++i) {
        for (int j = 0; j < num_dimensions; ++j) {
            agents[i][j] = search_space[j].first + (search_space[j].second - search_space[j].first) * static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    std::vector<double> fitness(num_agents);
    for (int i = 0; i < num_agents; ++i) {
        fitness[i] = benchmark_function(agents[i]);
    }

    // Выбор империй на основе приспособленности
    std::vector<int> sorted_indices(num_agents);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&fitness](int i1, int i2) { return fitness[i1] < fitness[i2]; });

    std::vector<std::vector<double>> empires(num_empires);
    std::vector<std::vector<double>> colonies(num_agents - num_empires);
    std::vector<double> empire_fitness(num_empires);
    std::vector<double> colony_fitness(num_agents - num_empires);

    for (int i = 0; i < num_empires; ++i) {
        empires[i] = agents[sorted_indices[i]];
        empire_fitness[i] = fitness[sorted_indices[i]];
    }
    for (int i = 0; i < colonies.size(); ++i) {
        colonies[i] = agents[sorted_indices[i + num_empires]];
        colony_fitness[i] = fitness[sorted_indices[i + num_empires]];
    }

    // Инициализация адаптивных скоростей обучения и коэффициентов ассимиляции
    double learning_rate_init = 0.5;
    double learning_rate_final = 0.01;
    double assimilation_coeff_init = 0.5;
    double assimilation_coeff_final = 0.1;

    for (int t = 0; t < max_iter; ++t) {
        double assimilation_coeff = assimilation_coeff_init - (assimilation_coeff_init - assimilation_coeff_final) * static_cast<double>(t) / max_iter;
        double learning_rate = learning_rate_init - (learning_rate_init - learning_rate_final) * static_cast<double>(t) / max_iter;

        // Осуществляем скрещивание между империями
        for (int i = 0; i < num_empires; ++i) {
            if (static_cast<double>(std::rand()) / RAND_MAX < 0.5) {
                int other = std::rand() % num_empires;
                std::vector<double> child(num_dimensions);
                for (int j = 0; j < num_dimensions; ++j) {
                    child[j] = 0.5 * (empires[i][j] + empires[other][j]);
                }
                double child_fitness = benchmark_function(child);
                if (child_fitness < empire_fitness[i]) {
                    empires[i] = child;
                    empire_fitness[i] = child_fitness;
                }
            }
        }

        // Обновление позиций колоний на основе их соответствующих империй
        for (int i = 0; i < colonies.size(); ++i) {
            for (int j = 0; j < num_dimensions; ++j) {
                colonies[i][j] -= learning_rate * assimilation_coeff * (colonies[i][j] - empires[i % num_empires][j]);
            }
        }

        // Осуществляем революцию, внося случайные возмущения
        for (int i = 0; i < colonies.size(); ++i) {
            for (int j = 0; j < num_dimensions; ++j) {
                colonies[i][j] += 0.2 * static_cast<double>(rand()) / RAND_MAX;
            }
        }

        // Обновление приспособленности всех агентов
        for (int i = 0; i < num_empires; ++i) {
            empire_fitness[i] = benchmark_function(empires[i]);
        }
        for (int i = 0; i < colonies.size(); ++i) {
            colony_fitness[i] = benchmark_function(colonies[i]);
        }

        // Поиск лучшей империи
        int best_index = std::distance(empire_fitness.begin(), std::min_element(empire_fitness.begin(), empire_fitness.end()));
        auto best_agent = empires[best_index];
        double best_fitness = empire_fitness[best_index];

        if (*std::min_element(empire_fitness.begin(), empire_fitness.end()) < 1e-9) {
            break;
        }
    }

    int best_index = std::distance(empire_fitness.begin(), std::min_element(empire_fitness.begin(), empire_fitness.end()));
    return {empires[best_index], empire_fitness[best_index]};
}

double mean(const std::vector<double>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

double stddev(const std::vector<double>& vec, double mean) {
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    double sq_mean = sq_sum / vec.size();
    return std::sqrt(sq_mean - mean * mean);
}

int main() {
    int num_agents = 100;
    int max_iter = 3000;
    std::vector<std::pair<double, double>> search_space(20, {-10.0, 10.0});
    int num_empires = 10;
    int num_runs = 30;

    // A vector of function pointers to the benchmark functions
    std::vector<double (*)(const std::vector<double>&)> benchmark_functions = {f1, f2, f3, f4, f5};

    for (auto benchmark_function : benchmark_functions) {
        std::vector<double> fitness_results;
        std::vector<double> run_times;
        for (int i = 0; i < num_runs; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();
            auto [best_agent, fitness] = ica(num_agents, max_iter, search_space, num_empires, benchmark_function);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> run_time = end_time - start_time;

            fitness_results.push_back(fitness);
            run_times.push_back(run_time.count());
        }

        double mean_fitness = mean(fitness_results);
        double fitness_std_dev = stddev(fitness_results, mean_fitness);
        double mean_run_time = mean(run_times);

        // Printing the results
        std::cout << "Benchmark function: " << (benchmark_function == f1 ? "f1" :
                                                benchmark_function == f2 ? "f2" :
                                                benchmark_function == f3 ? "f3" :
                                                benchmark_function == f4 ? "f4" : "f5") << std::endl;
        std::cout << "Average fitness: " << mean_fitness << std::endl;
        std::cout << "Standard deviation: " << fitness_std_dev << std::endl;
        std::cout << "Average run time: " << mean_run_time << " sec" << std::endl;
        std::cout << std::endl;
    }

    return 0;
}