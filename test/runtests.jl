using Test
using Random
using LinearAlgebra
# 引入我们刚才定义的模块
# include("./src/SparseRationalLearning.jl")
# using .SparseRationalLearning

using SparseRationalLearning

# 设置随机种子 [cite: 1]
Random.seed!(42)

# 定义测试用的黑盒函数 [cite: 12, 33]
# f(x1, x2) = (1.5*x1*x2 + 0.8*x1^2) / (1 + 0.2*x2^2)
function complex_blackbox(X)
    x1, x2 = X[:, 1], X[:, 2]
    return (1.5 .* x1 .* x2 .+ 0.8 .* x1.^2) ./ (1.0 .+ 0.2 .* x2.^2)
end

@testset "SparseRationalLearning Tests" begin
    # 生成测试数据
    n_samples = 150
    X_data = rand(n_samples, 2) .* 5.0
    # 添加少量噪声
    y_data = complex_blackbox(X_data) + randn(n_samples) * 0.05
    
    degree = 2
    truncation_threshold = 1e-2 #截断
	time_limit= 60.0 # 60秒, 优化算法时间限制，单位：秒

    @testset "L0 Algorithm (MIP)" begin
        println("\n" * "="^10 * " Testing L0 Algorithm " * "="^10)
        lambda_l0 = 0.5  # 该参数影响性能
        M_val = 50.0
        
        a, b, labels = train_l0(X_data, y_data, degree, lambda_l0; M=M_val, time_limit=60.0)
        
        @test a !== nothing
        @test b !== nothing
        
        if a !== nothing           
            # 打印表达式
            expr = sparse_rational_expression(a, b, labels, threshold=0.0)
            println("L0 Result Expression: \n  $expr")
            
            # 计算 R2
            r2, _ = calculate_r2(X_data, y_data, a, b, degree)
            println("L0 R²: $(round(r2, digits=4))")
            
            # 验证 R2 是否在合理范围内 (例如 > 0.9)
            @test r2 > 0.9
        end
    end

    @testset "L1 Algorithm (Lasso)" begin
        println("\n" * "="^10 * " Testing L1 Algorithm " * "="^10)
        lambda_l1 = 1.0  # 该参数影响性能
        
        a, b, labels = train_l1(X_data, y_data, degree, lambda_l1)
        
        @test a !== nothing
        @test b !== nothing
        
        if a !== nothing		    
			expr = sparse_rational_expression(a, b, labels, threshold=0.0)
            println("L1 Result Expression: \n  $expr")
			r2, _ = calculate_r2(X_data, y_data, a, b, degree)
            println("L1 R²: $(round(r2, digits=4))")
			@test r2 > 0.90
			
			@info("truncate, re-evaluate and display... ...")			
            a_clean, b_clean = coefficients_threshold(a, b, threshold=truncation_threshold)
            expr = sparse_rational_expression(a_clean, b_clean, labels, threshold=truncation_threshold)
            println("L1 Result Expression: \n  $expr")            
            r2, _ = calculate_r2(X_data, y_data, a_clean, b_clean, degree)
            println("L1 R²: $(round(r2, digits=4))")
            
            
        end
    end
end
