module SparseRationalLearning

using LinearAlgebra, Statistics, Combinatorics
using Convex
using HiGHS, SCS
import MathOptInterface as MOI # 引入 MOI 以支持传递求解器参数

export train_l0, train_l1, calculate_r2, sparse_rational_expression, coefficients_threshold

"""
    get_multivariate_basis(X, degree)

使用 Combinatorics 自动生成所有总次数 <= degree 的多项式基。
返回: 基矩阵 Phi, 以及描述每一项的索引列表。
"""
function get_multivariate_basis(X, degree)
    n_samples, n_vars = size(X)
    basis_columns = [ones(n_samples)] # 常数项 (degree 0)
    basis_labels = [Dict("powers" => zeros(Int, n_vars))] 
    
    # 遍历次数从 1 到 degree
    for d in 1:degree
        for combo in with_replacement_combinations(1:n_vars, d)
            col = ones(n_samples)
            powers = zeros(Int, n_vars)
            for var_idx in combo
                col .*= X[:, var_idx]
                powers[var_idx] += 1
            end
            push!(basis_columns, col)
            push!(basis_labels, Dict("powers" => powers))
        end
    end
    
    return hcat(basis_columns...), basis_labels
end

"""
    train_l0(X, y, degree, lambda; M=50.0, time_limit=60.0)

基于 L0 范数（混合整数规划 MIP）的稀疏有理回归算法。
模型: minimize ||P(x) - y*Q(x)||_1 + lambda * ||z||_0
求解器: HiGHS (支持设置 time_limit)
"""
function train_l0(X, y, degree, lambda; M=50.0, time_limit=60.0)
    Phi, basis_labels = get_multivariate_basis(X, degree)
    n_basis = size(Phi, 2)
    
    # 构建线性化系统: P - y*(Q-1) = y (注意 Q 的常数项设为 1)
    Phi_P = Phi
    Phi_Q_reduced = y .* Phi[:, 2:end]
    DesignMatrix = hcat(Phi_P, -Phi_Q_reduced)
    total_params = size(DesignMatrix, 2)

    # 定义优化变量
    coeffs = Variable(total_params)          
    z = Variable(total_params, BinVar)      

    # 构造约束条件：大M法
    constraints = [
        coeffs <= M * z,
        coeffs >= -M * z
    ]

    # 定义目标函数：L1 拟合误差 + L0 惩罚
    error_term = sum(abs(DesignMatrix * coeffs - y))
    penalty_term = lambda * sum(z)
    
    problem = minimize(error_term + penalty_term, constraints)

    # --- 修复点：使用 MOI.OptimizerWithAttributes 传递参数 ---
    optimizer = MOI.OptimizerWithAttributes(HiGHS.Optimizer, "time_limit" => time_limit)
    
    # 调用 HiGHS 求解器 (silent=true 是 Convex.solve! 支持的通用参数)
    solve!(problem, optimizer; silent = true)

    if problem.status == Convex.MOI.OPTIMAL || problem.status == Convex.MOI.ALMOST_OPTIMAL
        val_coeffs = evaluate(coeffs)
        a_recovered = val_coeffs[1:n_basis]
        b_recovered = vcat([1.0], val_coeffs[n_basis+1:end])
        return a_recovered, b_recovered, basis_labels
    else
        @warn "L0 优化未获得最优解，状态: $(problem.status)"
        return nothing, nothing, nothing
    end
end

"""
    train_l1(X, y, degree, lambda)

基于 L1 范数（Lasso）的稀疏有理回归算法。
模型: minimize ||P(x) - y*Q(x)||_2^2 + lambda * ||coeffs||_1
求解器: SCS
"""
function train_l1(X, y, degree, lambda)
    Phi, basis_labels = get_multivariate_basis(X, degree)
    n_basis = size(Phi, 2)

    # 构建线性化系统
    Phi_P = Phi
    Phi_Q_reduced = y .* Phi[:, 2:end]
    DesignMatrix = hcat(Phi_P, -Phi_Q_reduced)
    total_coeffs = size(DesignMatrix, 2)

    # 求解 L1 正则化问题
    coeffs = Variable(total_coeffs)
    # 目标函数：平方误差 + L1 范数
    problem = minimize(sumsquares(DesignMatrix * coeffs - y) + lambda * norm(coeffs, 1))
    
    # 调用 SCS 求解器
    solve!(problem, SCS.Optimizer; silent = true)

    if problem.status == Convex.MOI.OPTIMAL || problem.status == Convex.MOI.ALMOST_OPTIMAL
        val = evaluate(coeffs)
        a_recovered = val[1:n_basis]
        b_recovered = vcat([1.0], val[n_basis+1:end])
        return a_recovered, b_recovered, basis_labels
    else
        @warn "L1 优化失败，状态: $(problem.status)"
        return nothing, nothing, nothing
    end
end

"""
    format_poly(coeffs, labels; threshold=0.05)

辅助函数：将系数向量和标签转换为多项式字符串。
"""
function format_poly(coeffs, labels; threshold=0.0)
    terms = String[]
    for i in 1:length(coeffs)
        c = coeffs[i]
        if abs(c) > threshold
            powers = labels[i]["powers"]
            var_parts = String[]
            for (v_idx, p) in enumerate(powers)
                if p == 1
                    push!(var_parts, "x$v_idx")
                elseif p > 1
                    push!(var_parts, "x$v_idx^$p")
                end
            end
            var_str = join(var_parts, "*")
            c_round = round(c, digits=4)
            
            if isempty(var_str)
                push!(terms, "$c_round")
            else
                prefix = c_round == 1.0 ? "" : (c_round == -1.0 ? "-" : "$c_round*")
                push!(terms, prefix * var_str)
            end
        end
    end

    if isempty(terms)
        return "0"
    end

    res = join(terms, " + ")
    res = replace(res, "+ -" => "- ")
    return res
end

"""
    sparse_rational_expression(a, b, labels; threshold=0.01)

生成形如 (P(x)) / (Q(x)) 的完整有理函数表达式。
"""
function sparse_rational_expression(a, b, labels; threshold=0.0)
    num_str = format_poly(a, labels, threshold=threshold)
    # b 的第一个元素对应 labels[1] (常数项)
    den_str = format_poly(b, labels[1:length(b)], threshold=threshold)
    return "($num_str) / ($den_str)"
end

"""
    calculate_r2(X, y_true, a, b, degree)

计算模型的 R^2 决定系数。
"""
function calculate_r2(X, y_true, a, b, degree)
    Phi, _ = get_multivariate_basis(X, degree)
    # 防止除以零，虽然理论上优化过程会避免极小分母，但为了稳健性可加 epsilon
    denom = Phi[:, 1:length(b)] * b
    y_pred = (Phi * a) ./ denom
    
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - (ss_res / ss_tot), y_pred
end

"""
    coefficients_threshold(a, b; threshold=0.01)

将系数向量截断，小于阈值的置为 0.0。
"""
function coefficients_threshold(a::Vector{T}, b::Vector{T}; threshold=0.001) where T
    a_hold = [abs(val) < threshold ? 0.0 : val for val in a]
    b_hold = [abs(val) < threshold ? 0.0 : val for val in b]
    return a_hold, b_hold
end

end # module