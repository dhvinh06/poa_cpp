#include "POA.h"
#include "fun_info.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        const std::string function_name = (argc >= 2) ? argv[1] : "F1";
        const int search_agents = (argc >= 3) ? std::stoi(argv[2]) : 30;
        const int max_iterations = (argc >= 4) ? std::stoi(argv[3]) : 1000;
        const std::uint64_t seed = (argc >= 5) ? std::stoull(argv[4]) : 42ULL;

        seed_random(seed);

        const FunctionInfo info = fun_info(function_name);
        const POAResult result = POA(search_agents, max_iterations,
                                     info.lowerbound, info.upperbound,
                                     info.dimension, info.fitness);

        std::cout << std::setprecision(15);
        std::cout << "Function: " << function_name << '\n';
        std::cout << "Search agents: " << search_agents << '\n';
        std::cout << "Max iterations: " << max_iterations << '\n';
        std::cout << "Random seed: " << seed << '\n';
        std::cout << "Best position: [";
        for (std::size_t i = 0; i < result.best_position.size(); ++i) {
            if (i != 0) std::cout << ", ";
            std::cout << result.best_position[i];
        }
        std::cout << "]\n";
        std::cout << "Best objective value: " << result.best_score << '\n';
        std::cout << "Final stored convergence-curve value: "
                  << result.convergence_curve.back() << '\n';

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        std::cerr << "Usage: ./poa_cpp [F1..F23] [SearchAgents] [MaxIterations] [Seed]\n";
        return 1;
    }
}
