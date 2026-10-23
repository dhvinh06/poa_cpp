#include "POA.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

POAResult POA(int search_agents, int max_iterations,
              const std::vector<double> &lowerbound,
              const std::vector<double> &upperbound,
              int dimension,
              const ObjectiveFunction &fitness) {
    if (search_agents <= 0) throw std::invalid_argument("SearchAgents must be > 0");
    if (max_iterations <= 0) throw std::invalid_argument("Max_iterations must be > 0");
    if (dimension <= 0 || static_cast<int>(lowerbound.size()) != dimension ||
        static_cast<int>(upperbound.size()) != dimension) {
        throw std::invalid_argument("Bounds must each contain exactly dimension values");
    }

    std::vector<std::vector<double> > X(
        static_cast<std::size_t>(search_agents),
        std::vector<double>(static_cast<std::size_t>(dimension), 0.0));
    std::vector<double> fit(static_cast<std::size_t>(search_agents), 0.0);

    for (int j = 0; j < dimension; ++j) {
        for (int i = 0; i < search_agents; ++i) {
            X[i][j] = lowerbound[j] + random01() * (upperbound[j] - lowerbound[j]);
        }
    }

    for (int i = 0; i < search_agents; ++i) {
        fit[i] = fitness(X[i]);
    }

    double fbest = std::numeric_limits<double>::infinity();
    std::vector<double> Xbest(static_cast<std::size_t>(dimension), 0.0);
    std::vector<double> best_so_far(static_cast<std::size_t>(max_iterations), 0.0);

    for (int t = 1; t <= max_iterations; ++t) {
        const auto best_it = std::min_element(fit.begin(), fit.end());
        const double best = *best_it;
        const int location = static_cast<int>(std::distance(fit.begin(), best_it));
        if (t == 1) {
            Xbest = X[location];
            fbest = best;
        } else if (best < fbest) {
            fbest = best;
            Xbest = X[location];
        }

        const int k = random_integer(0, search_agents - 1);
        const std::vector<double> X_FOOD = X[k];
        const double F_FOOD = fit[k];

        for (int i = 0; i < search_agents; ++i) {
            const int I = static_cast<int>(std::round(1.0 + random01()));
            const double r = random01(); // one scalar shared across all dimensions
            std::vector<double> X_new(static_cast<std::size_t>(dimension));

            if (fit[i] > F_FOOD) {
                for (int j = 0; j < dimension; ++j) {
                    X_new[j] = X[i][j] + r * (X_FOOD[j] - I * X[i][j]);
                }
            } else {
                for (int j = 0; j < dimension; ++j) {
                    X_new[j] = X[i][j] + r * (X[i][j] - X_FOOD[j]);
                }
            }

            for (int j = 0; j < dimension; ++j) {
                X_new[j] = std::max(X_new[j], lowerbound[j]);
                X_new[j] = std::min(X_new[j], upperbound[j]);
            }

            const double f_new = fitness(X_new);
            if (f_new <= fit[i]) {
                X[i] = X_new;
                fit[i] = f_new;
            }

            // Phase 2
            const double radius = 0.2 * (1.0 - static_cast<double>(t) /
                                         static_cast<double>(max_iterations));
            for (int j = 0; j < dimension; ++j) {
                X_new[j] = X[i][j] + radius * (2.0 * random01() - 1.0) * X[i][j];
            }

            for (int j = 0; j < dimension; ++j) {
                X_new[j] = std::max(X_new[j], lowerbound[j]);
                X_new[j] = std::min(X_new[j], upperbound[j]);
            }

            const double f_new_phase2 = fitness(X_new);
            if (f_new_phase2 <= fit[i]) {
                X[i] = X_new;
                fit[i] = f_new_phase2;
            }
        }

        best_so_far[static_cast<std::size_t>(t - 1)] = fbest;
    }

    return {fbest, Xbest, best_so_far};
}
