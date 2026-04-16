# SparseRationalLearning

A Julia package for learning rational functions with sparsity from data.

## Introduction

`SparseRationalLearning` provides algorithms for learning rational functions of the form `P(x)/Q(x)` from data, where `P` and `Q` are polynomials. By imposing sparsity-promoting regularization, the package learns models that are both compact and interpretable.

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

# Generate sample data
X = rand(100, 2) .* 5.0 # 100 samples, 2 features
y = (1.5 .* X[:,1] .* X[:,2] .+ 0.8 .* X[:,1].^2) ./ (1.0 .+ 0.2 .* X[:,2].^2)

#  Train a sparse rational model using the L1-based method (fast)
a, b, labels = train_l1(X, y, 2, 1.0)

# Display the learned symbolic expression
println(sparse_rational_expression(a, b, labels, threshold=0.01))

# Compute the coefficient of determination (R²)
r2, _ = calculate_r2(X, y, a, b, 2)
println("R² = $r2")
```

## API Reference

### Training Functions

#### `train_l0(X, y, degree, lambda; M=50.0, time_limit=60.0)`

Fits a sparse rational function model using L0 regularization via mixed-integer programming.


**Parameters:**
- `X`: Feature matrix of size `(n_samples, n_features)`
- `y`: Target vector
- `degree`: Maximum polynomial degree
- `lambda`: Sparsity regularization parameter
- `M`: Big-M constant used in the mixed-integer formulation
- `time_limit`: Solver time limit in seconds

**Returns:** `(a, b, labels)` — numerator coefficients, denominator coefficients, and basis-function labels

---

#### `train_l1(X, y, degree, lambda)`

Fits a sparse rational function model using L1 regularization (Lasso).

**Parameters:**
- `X`: Feature matrix
- `y`: Target vector
- `degree`: Maximum polynomial degree
- `lambda`: L1 regularization coefficient

**Returns:** `(a, b, labels)` — numerator coefficients, denominator coefficients, and basis-function labels

---

### Utility Functions

| Function | Description |
|----------|-------------|
| `sparse_rational_expression(a, b, labels; threshold)` | Generate rational function expression string |
| `calculate_r2(X, y_true, a, b, degree)` | Calculate the coefficient of determination (R²) |
| `coefficients_threshold(a, b; threshold)` | Truncate small coefficients to 0 |

## Algorithm Comparison

| Feature | L0 Method | L1 Method |
|---------|-----------|-----------|
| Solver | HiGHS | SCS |
| Sparsity | More precise | Approximate |
| Speed | Slower | Faster |
| Recommended use | Small-scale problems | Large-scale problems |

## Running Tests

```julia
using Pkg
Pkg.test()
```

## Dependencies
- Julia >= 1.12
- Convex.jl - Convex optimization modeling
- HiGHS.jl - Mixed-integer programming (MIP) solver
- SCS.jl - Conic optimization solver
- Combinatorics.jl - Combinatorial utilities

## License
MIT License

## Citation

If you find this code or our methodology useful in your research, please consider citing our paper:

```bibtex
@article{li2026SpareRational,
  title={Sparse Rational Learning for Interpretable Aerodynamic Surrogate Modeling},
  author={Li, Jing and Luo, Changtong},
  journal={TBD},
  year={2026}
}
```

---

Co-author: [@lijing-creator](https://github.com/lijing-creator)

[中文文档](README_cn.md)
