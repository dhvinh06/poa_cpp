#include "fun_info.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace {
constexpr double PI = 3.141592653589793238462643383279502884;
std::mt19937_64 engine{std::random_device{}()};
std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

double sum_squares(const std::vector<double>& x) {
    double result = 0.0;
    for (double value : x) result += value * value;
    return result;
}

double Ufun(const std::vector<double>& x, double a, double k, int m) {
    double result = 0.0;
    for (double value : x) {
        if (value > a) {
            result += k * std::pow(value - a, m);
        } else if (value < -a) {
            result += k * std::pow(-value - a, m);
        }
    }
    return result;
}

FunctionInfo make_info(int dimension, double lower, double upper,
                       ObjectiveFunction fitness) {
    return {std::vector<double>(dimension, lower),
            std::vector<double>(dimension, upper), dimension,
            std::move(fitness)};
}

// F1
 double F1(const std::vector<double>& x) {
    return sum_squares(x);
}

// F2
 double F2(const std::vector<double>& x) {
    double sum_abs = 0.0;
    double product_abs = 1.0;
    for (double value : x) {
        const double a = std::abs(value);
        sum_abs += a;
        product_abs *= a;
    }
    return sum_abs + product_abs;
}

// F3
 double F3(const std::vector<double>& x) {
    double result = 0.0;
    double prefix_sum = 0.0;
    for (double value : x) {
        prefix_sum += value;
        result += prefix_sum * prefix_sum;
    }
    return result;
}

// F4
 double F4(const std::vector<double>& x) {
    double result = 0.0;
    for (double value : x) result = std::max(result, std::abs(value));
    return result;
}

// F5: Rosenbrock function
 double F5(const std::vector<double>& x) {
    double result = 0.0;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        const double a = x[i + 1] - x[i] * x[i];
        const double b = x[i] - 1.0;
        result += 100.0 * a * a + b * b;
    }
    return result;
}

// F6
 double F6(const std::vector<double>& x) {
    double result = 0.0;
    for (double value : x) {
        const double rounded_down = std::floor(value + 0.5);
        result += rounded_down * rounded_down;
    }
    return result;
}

// F7: includes the random term rand in the original MATLAB source.
 double F7(const std::vector<double>& x) {
    double result = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double index_1_based = static_cast<double>(i + 1);
        result += index_1_based * std::pow(x[i], 4.0);
    }
    return result + random01();
}

// F8
 double F8(const std::vector<double>& x) {
    double result = 0.0;
    for (double value : x) {
        result += -value * std::sin(std::sqrt(std::abs(value)));
    }
    return result;
}

// F9: Rastrigin function
 double F9(const std::vector<double>& x) {
    double result = 0.0;
    for (double value : x) {
        result += value * value - 10.0 * std::cos(2.0 * PI * value);
    }
    return result + 10.0 * static_cast<double>(x.size());
}

// F10: Ackley function
 double F10(const std::vector<double>& x) {
    const double dimension = static_cast<double>(x.size());
    double cosine_sum = 0.0;
    for (double value : x) cosine_sum += std::cos(2.0 * PI * value);
    return -20.0 * std::exp(-0.2 * std::sqrt(sum_squares(x) / dimension))
           - std::exp(cosine_sum / dimension) + 20.0 + std::exp(1.0);
}

// F11: Griewank function
 double F11(const std::vector<double>& x) {
    double square_sum = 0.0;
    double cosine_product = 1.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        square_sum += x[i] * x[i] / 4000.0;
        cosine_product *= std::cos(x[i] / std::sqrt(static_cast<double>(i + 1)));
    }
    return square_sum - cosine_product + 1.0;
}

// F12
 double F12(const std::vector<double>& x) {
    const std::size_t dimension = x.size();
    std::vector<double> y(dimension);
    for (std::size_t i = 0; i < dimension; ++i) y[i] = (x[i] + 1.0) / 4.0;

    double result = 10.0 * std::pow(std::sin(PI * (1.0 + y[0])), 2.0);
    for (std::size_t i = 0; i + 1 < dimension; ++i) {
        result += y[i] * y[i] *
                  (1.0 + 10.0 * std::pow(std::sin(PI * (1.0 + y[i + 1])), 2.0));
    }
    result += y[dimension - 1] * y[dimension - 1];
    return (PI / static_cast<double>(dimension)) * result + Ufun(x, 10.0, 100.0, 4);
}

// F13
 double F13(const std::vector<double>& x) {
    const std::size_t dimension = x.size();
    double result = std::pow(std::sin(3.0 * PI * x[0]), 2.0);
    for (std::size_t i = 0; i + 1 < dimension; ++i) {
        result += std::pow(x[i] - 1.0, 2.0) *
                  (1.0 + std::pow(std::sin(3.0 * PI * x[i + 1]), 2.0));
    }
    result += std::pow(x[dimension - 1] - 1.0, 2.0) *
              (1.0 + std::pow(std::sin(2.0 * PI * x[dimension - 1]), 2.0));
    return 0.1 * result + Ufun(x, 5.0, 100.0, 4);
}

// F14: Shekel's Foxholes function
 double F14(const std::vector<double>& x) {
    const std::array<double, 5> grid{-32.0, -16.0, 0.0, 16.0, 32.0};
    double reciprocal_sum = 0.0;
    for (std::size_t j = 0; j < 25; ++j) {
        const double a1 = grid[j % 5];
        const double a2 = grid[j / 5];
        const double d1 = x[0] - a1;
        const double d2 = x[1] - a2;
        const double b = std::pow(d1, 6.0) + std::pow(d2, 6.0);
        reciprocal_sum += 1.0 / (static_cast<double>(j + 1) + b);
    }
    return 1.0 / (1.0 / 500.0 + reciprocal_sum);
}

// F15: Kowalik function
 double F15(const std::vector<double>& x) {
    const std::array<double, 11> aK{
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
        0.0456, 0.0342, 0.0323, 0.0235, 0.0246};
    const std::array<double, 11> original_bK{
        0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};
    double result = 0.0;
    for (std::size_t i = 0; i < aK.size(); ++i) {
        const double b = 1.0 / original_bK[i];
        const double b2 = b * b;
        const double numerator = x[0] * (b2 + x[1] * b);
        const double denominator = b2 + x[2] * b + x[3];
        const double difference = aK[i] - numerator / denominator;
        result += difference * difference;
    }
    return result;
}

// F16: six-hump camel function
 double F16(const std::vector<double>& x) {
    const double x1 = x[0], x2 = x[1];
    return 4.0 * x1 * x1 - 2.1 * std::pow(x1, 4.0)
           + std::pow(x1, 6.0) / 3.0 + x1 * x2
           - 4.0 * x2 * x2 + 4.0 * std::pow(x2, 4.0);
}

// F17: Branin function
 double F17(const std::vector<double>& x) {
    const double x1 = x[0], x2 = x[1];
    const double term = x2 - x1 * x1 * 5.1 / (4.0 * PI * PI)
                        + 5.0 / PI * x1 - 6.0;
    return term * term + 10.0 * (1.0 - 1.0 / (8.0 * PI)) * std::cos(x1) + 10.0;
}

// F18: Goldstein-Price function
 double F18(const std::vector<double>& x) {
    const double x1 = x[0], x2 = x[1];
    const double a = 1.0 + std::pow(x1 + x2 + 1.0, 2.0) *
        (19.0 - 14.0 * x1 + 3.0 * x1 * x1 - 14.0 * x2
         + 6.0 * x1 * x2 + 3.0 * x2 * x2);
    const double b = 30.0 + std::pow(2.0 * x1 - 3.0 * x2, 2.0) *
        (18.0 - 32.0 * x1 + 12.0 * x1 * x1 + 48.0 * x2
         - 36.0 * x1 * x2 + 27.0 * x2 * x2);
    return a * b;
}

// F19: Hartmann 3-dimensional function
 double F19(const std::vector<double>& x) {
    const double aH[4][3] = {
        {3.0, 10.0, 30.0}, {0.1, 10.0, 35.0},
        {3.0, 10.0, 30.0}, {0.1, 10.0, 35.0}};
    const double cH[4] = {1.0, 1.2, 3.0, 3.2};
    const double pH[4][3] = {
        {0.3689, 0.117, 0.2673}, {0.4699, 0.4387, 0.747},
        {0.1091, 0.8732, 0.5547}, {0.03815, 0.5743, 0.8828}};
    double result = 0.0;
    for (int i = 0; i < 4; ++i) {
        double exponent_sum = 0.0;
        for (int j = 0; j < 3; ++j) {
            exponent_sum += aH[i][j] * std::pow(x[j] - pH[i][j], 2.0);
        }
        result -= cH[i] * std::exp(-exponent_sum);
    }
    return result;
}

// F20: Hartmann 6-dimensional function
 double F20(const std::vector<double>& x) {
    const double aH[4][6] = {
        {10.0, 3.0, 17.0, 3.5, 1.7, 8.0},
        {0.05, 10.0, 17.0, 0.1, 8.0, 14.0},
        {3.0, 3.5, 1.7, 10.0, 17.0, 8.0},
        {17.0, 8.0, 0.05, 10.0, 0.1, 14.0}};
    const double cH[4] = {1.0, 1.2, 3.0, 3.2};
    const double pH[4][6] = {
        {0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886},
        {0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991},
        {0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650},
        {0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381}};
    double result = 0.0;
    for (int i = 0; i < 4; ++i) {
        double exponent_sum = 0.0;
        for (int j = 0; j < 6; ++j) {
            exponent_sum += aH[i][j] * std::pow(x[j] - pH[i][j], 2.0);
        }
        result -= cH[i] * std::exp(-exponent_sum);
    }
    return result;
}

const double aSH[10][4] = {
    {4, 4, 4, 4}, {1, 1, 1, 1}, {8, 8, 8, 8}, {6, 6, 6, 6},
    {3, 7, 3, 7}, {2, 9, 2, 9}, {5, 5, 3, 3}, {8, 1, 8, 1},
    {6, 2, 6, 2}, {7, 3.6, 7, 3.6}};
const double cSH[10] = {0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5};

double shekel(const std::vector<double>& x, int count) {
    double result = 0.0;
    for (int i = 0; i < count; ++i) {
        double distance_squared = 0.0;
        for (int j = 0; j < 4; ++j) {
            const double difference = x[j] - aSH[i][j];
            distance_squared += difference * difference;
        }
        result -= 1.0 / (distance_squared + cSH[i]);
    }
    return result;
}

// F21, F22 and F23 are the first 5, first 7, and all 10 Shekel terms.
 double F21(const std::vector<double>& x) { return shekel(x, 5); }
 double F22(const std::vector<double>& x) { return shekel(x, 7); }
 double F23(const std::vector<double>& x) { return shekel(x, 10); }

} // namespace

void seed_random(std::uint64_t seed) {
    engine.seed(seed);
}

double random01() {
    return uniform_distribution(engine);
}

int random_integer(int low, int high) {
    std::uniform_int_distribution<int> distribution(low, high);
    return distribution(engine);
}

FunctionInfo fun_info(const std::string& F) {
    if (F == "F1")  return make_info(30, -100.0, 100.0, F1);
    if (F == "F2")  return make_info(30, -10.0, 10.0, F2);
    if (F == "F3")  return make_info(30, -100.0, 100.0, F3);
    if (F == "F4")  return make_info(30, -100.0, 100.0, F4);
    if (F == "F5")  return make_info(30, -30.0, 30.0, F5);
    if (F == "F6")  return make_info(30, -100.0, 100.0, F6);
    if (F == "F7")  return make_info(30, -1.28, 1.28, F7);
    if (F == "F8")  return make_info(30, -500.0, 500.0, F8);
    if (F == "F9")  return make_info(30, -5.12, 5.12, F9);
    if (F == "F10") return make_info(30, -32.0, 32.0, F10);
    if (F == "F11") return make_info(30, -600.0, 600.0, F11);
    if (F == "F12") return make_info(30, -50.0, 50.0, F12);
    if (F == "F13") return make_info(30, -50.0, 50.0, F13);
    if (F == "F14") return make_info(2, -65.536, 65.536, F14);
    if (F == "F15") return make_info(4, -5.0, 5.0, F15);
    if (F == "F16") return make_info(2, -5.0, 5.0, F16);
    if (F == "F17") return { {-5.0, 0.0}, {10.0, 15.0}, 2, F17 };
    if (F == "F18") return make_info(2, -2.0, 2.0, F18);
    if (F == "F19") return make_info(3, 0.0, 1.0, F19);
    if (F == "F20") return make_info(6, 0.0, 1.0, F20);
    if (F == "F21") return make_info(4, 0.0, 10.0, F21);
    if (F == "F22") return make_info(4, 0.0, 10.0, F22);
    if (F == "F23") return make_info(4, 0.0, 10.0, F23);

    throw std::invalid_argument("Function name must be F1 through F23; got: " + F);
}
