#ifndef POA_H
#define POA_H

#include <vector>

#include "fun_info.h"

struct POAResult {
    double best_score{};
    std::vector<double> best_position;
    std::vector<double> convergence_curve;
};

POAResult POA(int search_agents, int max_iterations,
              const std::vector<double>& lowerbound,
              const std::vector<double>& upperbound,
              int dimension,
              const ObjectiveFunction& fitness);

#endif
