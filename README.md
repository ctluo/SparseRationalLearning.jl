# SparseRationalLearning

A Julia package for sparse rational function regression learning.

## Introduction

This package implements algorithms to learn rational function expressions `P(x)/Q(x)` from data, where `P` and `Q` are polynomials. Through sparsity constraints, the learned expressions are kept as simple and interpretable as possible.

## Installation

```julia
# Navigate to the project directory and start Julia REPL
julia

# Activate the project environment
using Pkg
Pkg.activate(".")

# Install dependencies
Pkg.instantiate()
```

## Quick Start

```julia
using SparseRationalLearning

# Prepare data
X = rand(100, 2) .* 5.0# 100 samples, 2 features
y = (1.5 .* X[:,1] .* X[:,2] .+ 0.8 .* X[:,1].^2) ./ (1.0 .+ 0.2 .* X[:,2].^2)

# Train using L1 method (fast)
a, b, labels = train_l1(X, y, 2, 1.0)

# View the resulting expression
println(sparse_rational_expression(a, b, labels, threshold=0.01))

# Calculate R² score
r2, _ = calculate_r2(X, y, a, b, 2)
println("R² = $r2")
```

## API Reference

### Training Functions

#### `train_l0(X, y, degree, lambda; M=50.0, time_limit=60.0)`

Sparse rational regression based on L0 norm (Mixed Integer Programming).

**Parameters:**
- `X`: Feature matrix (n_samples × n_features)
- `y`: Target vector
- `degree`: Maximum polynomial degree
- `lambda`: Sparsity regularization coefficient
- `M`: Big-M parameter
- `time_limit`: Solver time limit (seconds)

**Returns:** `(a, b, labels)` - numerator coefficients, denominator coefficients, basis function labels

---

#### `train_l1(X, y, degree, lambda)`

Sparse rational regression based on L1 norm (Lasso).

**Parameters:**
- `X`: Feature matrix
- `y`: Target vector
- `degree`: Maximum polynomial degree
- `lambda`: L1 regularization coefficient

**Returns:** `(a, b, labels)`

---

### Utility Functions

| Function | Description |
|----------|-------------|
| `sparse_rational_expression(a, b, labels; threshold)` | Generate rational function expression string |
| `calculate_r2(X, y_true, a, b, degree)` | Calculate R² coefficient of determination |
| `coefficients_threshold(a, b; threshold)` | Truncate small coefficients to 0 |

## Algorithm Comparison

| Feature | L0 Method | L1 Method |
|---------|-----------|-----------|
| Solver | HiGHS | SCS |
| Sparsity | More precise | Approximate |
| Speed | Slower | Fast |
| Use Case | Small-scale problems | Large-scale problems |

## Running Tests

```julia
using Pkg
Pkg.test()
```

## Dependencies

- Julia >= 1.11
- Convex.jl - Convex optimization modeling
- HiGHS.jl - MIP solver
- SCS.jl - Conic solver
- Combinatorics.jl - Combinatorics utilities

## License

MIT License

---

[中文文档](README_cn.md)
