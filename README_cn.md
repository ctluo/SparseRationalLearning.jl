# SparseRationalLearning

一个用于稀疏有理函数回归学习的 Julia 包。

## 简介

本包实现了从数据中学习有理函数表达式 `P(x)/Q(x)` 的算法，其中 `P` 和 `Q` 是多项式。通过稀疏性约束，使得学习到的表达式尽可能简洁、可解释。

## 安装

```julia
# 进入项目目录后，启动 Julia REPL
julia

# 激活项目环境
using Pkg
Pkg.activate(".")

# 安装依赖
Pkg.instantiate()
```

## 快速开始

```julia
using SparseRationalLearning

# 准备数据
X = rand(100, 2) .* 5.0# 100 个样本，2 个特征
y = (1.5 .* X[:,1] .* X[:,2] .+ 0.8 .* X[:,1].^2) ./ (1.0 .+ 0.2 .* X[:,2].^2)

# 使用 L1 方法训练 (快速)
a, b, labels = train_l1(X, y, 2, 1.0)

# 查看结果表达式
println(sparse_rational_expression(a, b, labels, threshold=0.01))

# 计算 R² 评分
r2, _ = calculate_r2(X, y, a, b, 2)
println("R² = $r2")
```

## API 文档

### 训练函数

#### `train_l0(X, y, degree, lambda; M=50.0, time_limit=60.0)`

基于 L0 范数的稀疏有理回归（混合整数规划）。

**参数：**
- `X`: 特征矩阵 (n_samples × n_features)
- `y`: 目标向量
- `degree`: 多项式最高次数
- `lambda`: 稀疏正则化系数
- `M`: 大M参数
- `time_limit`: 求解时间限制（秒）

**返回：** `(a, b, labels)` - 分子系数、分母系数、基函数标签

---

#### `train_l1(X, y, degree, lambda)`

基于 L1 范数的稀疏有理回归（Lasso）。

**参数：**
- `X`: 特征矩阵
- `y`: 目标向量
- `degree`: 多项式最高次数
- `lambda`: L1 正则化系数

**返回：** `(a, b, labels)`

---

### 辅助函数

| 函数 | 描述 |
|------|------|
| `sparse_rational_expression(a, b, labels; threshold)` | 生成有理函数表达式字符串 |
| `calculate_r2(X, y_true, a, b, degree)` | 计算 R² 决定系数 |
| `coefficients_threshold(a, b; threshold)` | 截断小系数为 0 |

## 算法对比

| 特性 | L0 方法 | L1 方法 |
|------|---------|---------|
| 求解器 | HiGHS | SCS |
| 稀疏性 | 更精确 | 近似 |
| 计算速度 | 较慢 | 快 |
| 适用场景 | 小规模问题 | 大规模问题 |

## 运行测试

```julia
using Pkg
Pkg.test()
```

## 依赖

- Julia >= 1.11
- Convex.jl - 凸优化建模
- HiGHS.jl - MIP 求解器
- SCS.jl - 锥规划求解器
- Combinatorics.jl - 组合数学工具

## 许可证

MIT License
