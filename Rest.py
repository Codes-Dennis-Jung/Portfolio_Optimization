








#### Backtest ####

# Example usage
def run_backtest_example():
    """Run backtest example with multiple strategies"""
    # Get test data
    returns, stocks, benchmark_returns = get_test_data()
    
    # Initialize backtest optimizer
    backtest_optimizer = RobustBacktestOptimizer(
        returns=returns,
        uncertainty=0.1,
        risk_aversion=0.5,
        lookback_window=252*3,  # 1 year
        rebalance_frequency=21  # Monthly
    )
    
    # Define strategies to test
    strategies = {
        "Robust Mean-Variance": (
            ObjectiveFunction.ROBUST_MEAN_VARIANCE,
            OptimizationConstraints(
                long_only=True,
                box_constraints={i: (0.0, 0.5) for i in range(len(returns.columns))}
            ),
            {'epsilon': 0.1, 'kappa': 0.05}
        ),
        "Risk Parity": (
            ObjectiveFunction.RISK_PARITY,
            OptimizationConstraints(long_only=True),
            {}
        )
    }
    
    # Run backtests
    results = {}
    for name, (objective, constraints, kwargs) in strategies.items():
        print(f"\nBacktesting {name} strategy...")
        try:
            result = backtest_optimizer.run_backtest(
                objective=objective,
                constraints=constraints,
                **kwargs
            )
            results[name] = result
            
            # Print metrics
            print("\nBacktest Metrics:")
            print(result['backtest_metrics'])
            
            # Plot results
            backtest_optimizer.plot_backtest_results(result)
            
        except Exception as e:
            print(f"Backtest failed for {name}: {str(e)}")
    
    return results

def run_cvxpy_optimization_example():
    """Run comprehensive example using CVXPY optimizer"""
    
    print("Running CVXPY Portfolio Optimization Examples...")
    
    # 1. Get test data
    returns, stocks, benchmark_returns = get_test_data()
    
    # 2. Initialize optimizer with CVXPY method
    optimizer = RobustPortfolioOptimizer(
        returns=returns,
        optimization_method=OptimizationMethod.CVXPY,  # Specify CVXPY
        uncertainty=0.1,
        risk_aversion=1.0,
        half_life=36,
        risk_free_rate=0.02,
        transaction_cost=0.001
    )
    
    # 3. Run different optimization scenarios
    results = {}
    
    # 3.1 Minimum Variance with Sector Constraints
    print("\n1. Minimum Variance with Sector Constraints")
    try:
        sector_constraints = {
            f"Sector_{i}": GroupConstraint(
                assets=list(range(i*3, (i+1)*3)),
                bounds=(0.1, 0.4)  # 10-40% per sector
            ) for i in range(4)
        }
        
        min_var_result = optimizer.optimize(
            objective=ObjectiveFunction.MINIMUM_VARIANCE,
            constraints=OptimizationConstraints(
                long_only=True,
                group_constraints=sector_constraints
            )
        )
        results["Minimum Variance"] = min_var_result
        print_detailed_metrics(min_var_result, "Minimum Variance")
        
    except Exception as e:
        print(f"Minimum Variance optimization failed: {e}")
    
    # 3.2 Robust Mean-Variance with Box Constraints
    print("\n2. Robust Mean-Variance with Box Constraints")
    try:
        robust_mv_result = optimizer.optimize(
            objective=ObjectiveFunction.ROBUST_MEAN_VARIANCE,
            constraints=OptimizationConstraints(
                long_only=True,
                box_constraints={i: (0.02, 0.15) for i in range(len(returns.columns))}
            ),
            epsilon=0.1,
            kappa=1.0
        )
        results["Robust Mean-Variance"] = robust_mv_result
        print_detailed_metrics(robust_mv_result, "Robust Mean-Variance")
        
    except Exception as e:
        print(f"Robust Mean-Variance optimization failed: {e}")
    
    # 3.3 Maximum Sharpe with Turnover Constraint
    print("\n3. Maximum Sharpe with Turnover Constraint")
    try:
        max_sharpe_result = optimizer.optimize(
            objective=ObjectiveFunction.MAXIMUM_SHARPE,
            constraints=OptimizationConstraints(
                long_only=True,
                max_turnover=0.5
            )
        )
        results["Maximum Sharpe"] = max_sharpe_result
        print_detailed_metrics(max_sharpe_result, "Maximum Sharpe")
        
    except Exception as e:
        print(f"Maximum Sharpe optimization failed: {e}")
    
    # 3.4 CVaR Optimization with Scenarios
    print("\n4. CVaR Optimization with Scenarios")
    try:
        # Generate stress scenarios
        scenarios = generate_scenarios(returns)
        
        cvar_result = optimizer.optimize(
            objective=ObjectiveFunction.MINIMUM_CVAR,
            constraints=OptimizationConstraints(
                long_only=True,
                target_return=0.10  # 10% target return
            ),
            alpha=0.05,
            scenarios=scenarios
        )
        results["CVaR"] = cvar_result
        print_detailed_metrics(cvar_result, "CVaR")
        
    except Exception as e:
        print(f"CVaR optimization failed: {e}")
    
    # 3.5 Maximum Diversification with Long-Short
    print("\n5. Maximum Diversification with Long-Short")
    try:
        max_div_result = optimizer.optimize(
            objective=ObjectiveFunction.MAXIMUM_DIVERSIFICATION,
            constraints=OptimizationConstraints(
                long_only=False,
                box_constraints={i: (-0.2, 0.2) for i in range(len(returns.columns))}
            )
        )
        results["Maximum Diversification"] = max_div_result
        print_detailed_metrics(max_div_result, "Maximum Diversification")
        
    except Exception as e:
        print(f"Maximum Diversification optimization failed: {e}")
    
    # 4. Compare Results
    compare_optimization_results(results, returns.columns)
    
    return results

def compare_optimization_results(results: Dict, asset_names: pd.Index):
    """Create comprehensive comparison of optimization results"""
    # Create comparison DataFrames
    metrics_comparison = pd.DataFrame({
        name: {
            'Return': result['return'],
            'Risk': result['risk'],
            'Sharpe Ratio': result['sharpe_ratio'],
            'Turnover': result['turnover']
        }
        for name, result in results.items()
    }).T
    
    weights_comparison = pd.DataFrame({
        name: result['weights']
        for name, result in results.items()
    }, index=asset_names).T
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2)
    
    # 1. Risk-Return Plot
    ax1 = fig.add_subplot(gs[0, 0])
    plot_risk_return_comparison(metrics_comparison, ax1)
    
    # 2. Weights Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    plot_weights_comparison(weights_comparison, ax2)
    
    # 3. Metrics Comparison
    ax3 = fig.add_subplot(gs[1, :])
    plot_metrics_comparison(metrics_comparison, ax3)
    
    # 4. Sector Allocation Comparison
    ax4 = fig.add_subplot(gs[2, :])
    plot_sector_allocation_comparison(weights_comparison, ax4)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed comparison
    print("\nDetailed Metrics Comparison:")
    print(metrics_comparison.round(4))
    print("\nWeight Allocations:")
    print(weights_comparison.round(4))

def plot_risk_return_comparison(metrics: pd.DataFrame, ax: plt.Axes):
    """Plot risk-return comparison"""
    ax.scatter(metrics['Risk'], metrics['Return'], s=1000, alpha=0.6)
    
    for idx, row in metrics.iterrows():
        ax.annotate(idx, (row['Risk'], row['Return']))
        
    ax.set_xlabel('Risk')
    ax.set_ylabel('Expected Return')
    ax.set_title('Risk-Return Comparison')
    ax.grid(True)

def plot_weights_comparison(weights: pd.DataFrame, ax: plt.Axes):
    """Plot weight comparison"""
    weights.plot(kind='bar', ax=ax)
    ax.set_title('Weight Allocation Comparison')
    ax.set_ylabel('Weight')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

def plot_metrics_comparison(metrics: pd.DataFrame, ax: plt.Axes):
    """Plot metrics comparison"""
    metrics.plot(kind='bar', ax=ax)
    ax.set_title('Performance Metrics Comparison')
    ax.set_ylabel('Value')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

def plot_sector_allocation_comparison(weights: pd.DataFrame, ax: plt.Axes):
    """Plot sector allocation comparison"""
    # Group weights by sector (assuming 3 assets per sector)
    sector_weights = pd.DataFrame()
    for i in range(0, weights.shape[1], 3):
        sector = f"Sector {i//3 + 1}"
        sector_weights[sector] = weights.iloc[:, i:i+3].sum(axis=1)
    
    sector_weights.plot(kind='bar', ax=ax, stacked=True)
    ax.set_title('Sector Allocation Comparison')
    ax.set_ylabel('Weight')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Run backtest with CVXPY
def run_cvxpy_backtest_example():
    """Run backtest using CVXPY optimizer"""
    # Get test data
    returns, stocks, benchmark_returns = get_test_data()
    
    # Initialize backtest optimizer with CVXPY
    backtest_optimizer = RobustBacktestOptimizer(
        returns=returns,
        optimization_method=OptimizationMethod.CVXPY,
        uncertainty=0.1,
        risk_aversion=1.0,
        lookback_window=252,
        rebalance_frequency=21
    )
    
    # Define strategies
    strategies = {
        "CVXPY Robust Mean-Variance": (
            ObjectiveFunction.ROBUST_MEAN_VARIANCE,
            OptimizationConstraints(
                long_only=True,
                box_constraints={i: (0.0, 0.3) for i in range(len(returns.columns))}
            ),
            {'epsilon': 0.1, 'kappa': 1.0}
        ),
        "CVXPY CVaR": (
            ObjectiveFunction.MINIMUM_CVAR,
            OptimizationConstraints(
                long_only=True,
                target_return=0.08
            ),
            {'alpha': 0.05}
        )
    }
    
    # Run backtests
    results = {}
    for name, (objective, constraints, kwargs) in strategies.items():
        print(f"\nBacktesting {name} strategy...")
        try:
            result = backtest_optimizer.run_backtest(
                objective=objective,
                constraints=constraints,
                **kwargs
            )
            results[name] = result
            
            # Print metrics
            print("\nBacktest Metrics:")
            print(result['backtest_metrics'])
            
            # Plot results
            backtest_optimizer.plot_backtest_results(result)
            
        except Exception as e:
            print(f"Backtest failed for {name}: {str(e)}")
    
    return results

if __name__ == "__main__":
    # Run CVXPY optimization examples
    print("Running CVXPY Optimization Examples...")
    optimization_results = run_cvxpy_optimization_example()
    
    print("\nRunning CVXPY Backtest Examples...")
    backtest_results = run_cvxpy_backtest_example()
    
    print("\nRunning Normal Portfolio Strategies...")
    # Portolio Strategy test
    results = run_portfolio_optimization_examples()
    
    print("\nRunning all tests...")
    # Run all tests
    portfolio_results, frontier_results = run_comprehensive_tests()
    
    print("\nRunning efficient frontier...")
    # Print frontier metrics
    print_frontier_metrics(frontier_results)
    
    print("\nRunning backtest with scipy...")
    # Run backtest
    results = run_backtest_example()