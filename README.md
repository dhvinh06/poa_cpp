# Pelican Optimization Algorithm (POA) — C++ port

This project ports the three MATLAB source files supplied in the conversation:

- `main.m` -> `main.cpp`
- `POA.m` -> `POA.cpp` and `POA.h`
- `fun_info.m` -> `fun_info.cpp` and `fun_info.h`

All 23 benchmark functions (F1–F23), bounds, dimensions, and the update steps in the supplied MATLAB `POA.m` are implemented.

## Build

With a C++17 compiler (g++ / MinGW on Windows):

```bash
g++ -std=c++17 -O2 main.cpp POA.cpp fun_info.cpp -o poa_cpp
```

On Windows with MinGW, the executable can be named `poa_cpp.exe`.

## Run

Defaults reproduce the MATLAB main settings (`F1`, 30 search agents, 1000 iterations):

```bash
./poa_cpp
```

Choose a function, population size, iteration count, and random seed:

```bash
./poa_cpp F9 30 1000 42
./poa_cpp F17 30 1000 123
```

The default seed is 42 to make C++ runs repeatable. MATLAB and C++ use different random-number implementations, so equal seeds do not imply identical numerical outputs.

## Fidelity notes

This version intentionally follows the supplied MATLAB source, not a repaired interpretation of the paper. In particular, it preserves:

1. Prey selection via a randomly selected existing population member (`X_FOOD = X(k,:)`).
2. Bound clamping after each phase.
3. Acceptance condition `f_new <= fit(i)`.
4. Updating/storing the best score at the start of an iteration, which can omit an improvement found in the final iteration from the returned best position/score and convergence curve.
5. The random noise term in F7 (`+ rand`).

The implementation preserves the algorithm's behavior and formulas, but exact bit-for-bit agreement with MATLAB is not expected because MATLAB and C++ have different random generators and floating-point evaluation details.
