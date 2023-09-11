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

// Salp Swarm Algorithm (SSA)
std::pair<std::vector<double>, double> ssa(int num_salps, int max_iter, const std::vector<std::pair<double, double>>& search_space,
                                           double (*benchmark_function)(const std::vector<double>&)) {
    int num_dimensions = search_space.size();
    std::vector<std::vector<double>> salps(num_salps, std::vector<double>(num_dimensions));
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Initialize salp positions randomly within the search space
    for (int i = 0; i < num_salps; ++i) {
        for (int j = 0; j < num_dimensions; ++j) {
            salps[i][j] = search_space[j].first + (search_space[j].second - search_space[j].first) * static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    // Calculate fitness for each salp
    std::vector<double> fitness(num_salps);
    for (int i = 0; i < num_salps; ++i) {
        fitness[i] = benchmark_function(salps[i]);
    }

    for (int t = 0; t < max_iter; ++t) {
        // Get the best salp
        int best_index = std::distance(fitness.begin(), std::min_element(fitness.begin(), fitness.end()));
        std::vector<double> best_salp = salps[best_index];

        // Update positions with adaptive parameter
        double w = 1.0 - (static_cast<double>(t) / max_iter);

        for (int i = 0; i < num_salps; ++i) {
            for (int j = 0; j < num_dimensions; ++j) {
                if (i == 0) {
                    salps[i][j] = best_salp[j]; // The first salp follows the lead
                } else {
                    // Subsequent salps follow their predecessor
                    salps[i][j] = (salps[i][j] + salps[i - 1][j]) / 2;

                    // Introduce randomization for the latter half of iterations
                    if (t > max_iter / 2) {
                        salps[i][j] += w * (2.0 * static_cast<double>(std::rand()) / RAND_MAX - 1.0); // random value in [-1,1]
                    }

                    // Boundary check
                    if (salps[i][j] < search_space[j].first) {
                        salps[i][j] = search_space[j].first;
                    } else if (salps[i][j] > search_space[j].second) {
                        salps[i][j] = search_space[j].second;
                    }
                }
            }
        }

        // Update fitness values
        for (int i = 0; i < num_salps; ++i) {
            fitness[i] = benchmark_function(salps[i]);
        }

        if (*std::min_element(fitness.begin(), fitness.end()) < 1e-9) {
            break;
        }
    }

    int best_index = std::distance(fitness.begin(), std::min_element(fitness.begin(), fitness.end()));
    return {salps[best_index], fitness[best_index]};
}

// Calculate mean of a vector
double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

// Calculate standard deviation of a vector given its mean
double stddev(const std::vector<double>& values, double mean_val) {
    double variance = std::accumulate(values.begin(), values.end(), 0.0, [mean_val](double acc, double val) {
        return acc + (val - mean_val) * (val - mean_val);
    }) / values.size();
    return std::sqrt(variance);
}

int main() {
    int num_salps = 100;
    int max_iter = 3000;
    std::vector<std::pair<double, double>> search_space(20, {-10.0, 10.0});
    int num_runs = 30;

    std::vector<double (*)(const std::vector<double>&)> benchmark_functions = {f1, f2, f3, f4, f5};
    std::vector<std::string> function_names = {"f1", "f2", "f3", "f4", "f5"};

    for (size_t func_idx = 0; func_idx < benchmark_functions.size(); ++func_idx) {
        auto& benchmark_function = benchmark_functions[func_idx];
        std::vector<double> fitness_results;
        std::vector<double> run_times;

        // Run SSA for each benchmark function
        for (int i = 0; i < num_runs; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();
            auto [best_salp, fitness] = ssa(num_salps, max_iter, search_space, benchmark_function);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> run_time = end_time - start_time;

            fitness_results.push_back(fitness);
            run_times.push_back(run_time.count());
        }

        double mean_fitness = mean(fitness_results);
        double fitness_std_dev = stddev(fitness_results, mean_fitness);
        double mean_run_time = mean(run_times);

        // Display results for each benchmark function
        std::cout << "Benchmark function: " << function_names[func_idx] << std::endl;
        std::cout << "Average fitness: " << mean_fitness << std::endl;
        std::cout << "Standard deviation: " << fitness_std_dev << std::endl;
        std::cout << "Average run time: " << mean_run_time << " sec" << std::endl;
        std::cout << "---------------------------------------------\n";
        std::cout << std::endl;
    }

    return 0;
}
