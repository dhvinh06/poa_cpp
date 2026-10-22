#ifndef FUN_INFO_H
#define FUN_INFO_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using ObjectiveFunction = std::function<double(const std::vector<double>&)>;

struct FunctionInfo {
    std::vector<double> lowerbound;
    std::vector<double> upperbound;
    int dimension{};
    ObjectiveFunction fitness;
};

// Equivalent to the MATLAB fun_info(F), for F1 through F23.
FunctionInfo fun_info(const std::string& F);

// Shared random generator used by POA and F7 (F7 contains a random noise term).
void seed_random(std::uint64_t seed);
double random01();                 // uniform random number in [0, 1)
int random_integer(int low, int high); // inclusive endpoints

#endif
