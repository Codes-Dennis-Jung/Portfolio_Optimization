import pandas as pd
import numpy as np
from PortOpt import *
from typing import Dict, List, Optional, Tuple, Union, Callable

################################################################################################################
#### Testing Portfolio Optimization
################################################################################################################

# Loading test data 
def get_test_data():
    """Get stock data from Yahoo Finance for testing"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Define stock universe by sector
    stocks = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL'],
        'Finance': ['JPM', 'BAC', 'GS'],
        'Healthcare': ['JNJ', 'PFE', 'UNH'],
        'Consumer': ['PG', 'KO', 'WMT']
    }
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6*365)  # 3 years of data
    
    # Download data
    prices = pd.DataFrame()
    failed_downloads = []
    
    print("Downloading stock data...")
    for sector, symbols in stocks.items():
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(data) > 0:
                    prices[symbol] = data['Adj Close']
                else:
                    failed_downloads.append(symbol)
            except Exception as e:
                print(f"Failed to download {symbol}: {e}")
                failed_downloads.append(symbol)
    
    if failed_downloads:
        print(f"Warning: Failed to download data for: {', '.join(failed_downloads)}")
    
    prices = prices.resample('M').ffill()
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Handle missing data and outliers
    returns = returns.fillna(method='ffill').fillna(method='bfill')
    
    # Remove extreme outliers (more than 5 standard deviations)
    for col in returns.columns:
        mean = returns[col].mean()
        std = returns[col].std()
        returns[col] = returns[col].clip(mean - 5*std, mean + 5*std)
    
    # Get benchmark data (S&P 500)
    try:
        benchmark = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        benchmark_returns = benchmark['Adj Close'].pct_change().dropna()
    except:
        print("Warning: Failed to download benchmark data")
        benchmark_returns = pd.Series(index=returns.index, data=0)  # Fallback to zeros
    
    print(f"Successfully downloaded data for {len(returns.columns)} stocks")
    print(f"Data shape: {returns.shape}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    return returns, stocks, benchmark_returns

def generate_scenarios(returns: pd.DataFrame, n_scenarios: int = 1000) -> np.ndarray:
    """Generate stress test scenarios using bootstrap with increased volatility"""
    n_samples, n_assets = returns.shape
    scenarios = np.zeros((n_scenarios, n_assets))
    
    # Generate scenarios
    for i in range(n_scenarios):
        # Sample returns with replacement
        sample_idx = np.random.choice(n_samples, size=n_samples)
        sampled_returns = returns.iloc[sample_idx].values
        
        # Increase volatility for stress scenario
        stress_factor = np.random.uniform(1.5, 2.5)
        scenarios[i] = sampled_returns.mean(axis=0) * stress_factor
    
    return scenarios


def create_portfolio_strategies(n_assets: int, returns: pd.DataFrame) -> Dict:
    """Create different portfolio strategies with their constraints"""
    
    # Create sector groupings
    sector_constraints = {
        f"Sector_{i}": GroupConstraint(
            assets=list(range(i*3, (i+1)*3)),
            bounds=(0.1, 0.4)
        ) for i in range(4)
    }
    
    # Generate stress scenarios
    scenarios = generate_scenarios(returns)
    
    strategies = {
        "1. Minimum Variance (Long-Only)": (
            ObjectiveFunction.MINIMUM_VARIANCE,
            OptimizationConstraints(long_only=True),
            {}
        ),
        
        "2. Robust Mean-Variance": (
            ObjectiveFunction.ROBUST_MEAN_VARIANCE,
            OptimizationConstraints(
                long_only=True,
                target_risk=0.15
            ),
            {'epsilon': 0.1, 'kappa': 1.0}
        ),
        
        "3. Maximum Sharpe with Sectors": (
            ObjectiveFunction.MAXIMUM_SHARPE,
            OptimizationConstraints(
                long_only=True,
                group_constraints=sector_constraints
            ),
            {}
        ),
        
        "4. Risk Parity with Bounds": (
            ObjectiveFunction.RISK_PARITY,
            OptimizationConstraints(
                long_only=True,
                box_constraints={i: (0.02, 0.20) for i in range(n_assets)}
            ),
            {}
        ),
        
        "5. Mean-CVaR with Scenarios": (
            ObjectiveFunction.MEAN_CVAR,
            OptimizationConstraints(long_only=True),
            {
                'alpha': 0.05,
                'lambda_cvar': 0.7,
                'scenarios': scenarios
            }
        ),
        
        "6. Maximum Diversification (Long-Short)": (
            ObjectiveFunction.MAXIMUM_DIVERSIFICATION,
            OptimizationConstraints(
                long_only=False,
                box_constraints={i: (-0.2, 0.2) for i in range(n_assets)}
            ),
            {}
        ),
        
        "7. Hierarchical Risk Parity": (
            ObjectiveFunction.HIERARCHICAL_RISK_PARITY,
            OptimizationConstraints(long_only=True),
            {'clusters': HierarchicalRiskParity.get_clusters(returns.values, n_clusters=4)}
        ),
        
        "8. Equal Risk Contribution": (
            ObjectiveFunction.EQUAL_RISK_CONTRIBUTION,
            OptimizationConstraints(
                long_only=True,
                max_turnover=0.5
            ),
            {}
        )
    }
    
    return strategies


def risk_return_scatter(results, ax):
    """Plot risk-return scatter with Sharpe ratio bubbles"""
    risks = [r['risk'] for r in results.values() if r is not None]
    returns = [r['return'] for r in results.values() if r is not None]
    sharpes = [r['sharpe_ratio'] for r in results.values() if r is not None]
    labels = [name for name, r in results.items() if r is not None]
    
    sc = ax.scatter(risks, returns, s=np.array(sharpes)*1000, alpha=0.6)
    
    for i, label in enumerate(labels):
        ax.annotate(label, (risks[i], returns[i]))
    
    ax.set_xlabel('Risk')
    ax.set_ylabel('Expected Return')
    ax.set_title('Risk-Return Trade-off')
    ax.grid(True)

def weight_distribution_plot(results, asset_names, ax):
    """Plot weight distributions for different strategies"""
    data = {name: result['weights'] for name, result in results.items() if result is not None}
    df = pd.DataFrame(data, index=asset_names)
    
    df.boxplot(ax=ax)
    ax.set_xticklabels(df.columns, rotation=45)
    ax.set_title('Weight Distribution by Strategy')
    ax.set_ylabel('Weight')

def sector_allocation_plot(results, stocks, ax):
    """Plot sector allocations for each strategy"""
    sector_allocations = {}
    for strategy, result in results.items():
        if result is not None:
            weights = result['weights']
            sector_alloc = {}
            start_idx = 0
            for sector, symbols in stocks.items():
                sector_alloc[sector] = weights[start_idx:start_idx + len(symbols)].sum()
                start_idx += len(symbols)
            sector_allocations[strategy] = sector_alloc
    
    df = pd.DataFrame(sector_allocations)
    df.plot(kind='bar', ax=ax)
    ax.set_title('Sector Allocations by Strategy')
    ax.set_ylabel('Allocation')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def robust_metrics_plot(results, ax):
    """Plot comparison of robust metrics"""
    metrics = {
        'Sharpe Ratio': [r['sharpe_ratio'] for r in results.values() if r is not None],
        'Worst-Case Return': [r.get('worst_case_return', 0) for r in results.values() if r is not None],
        'Diversification Ratio': [r.get('diversification_ratio', 0) for r in results.values() if r is not None]
    }
    
    df = pd.DataFrame(metrics, index=[name for name, r in results.items() if r is not None])
    df.plot(kind='bar', ax=ax)
    ax.set_title('Robust Metrics Comparison')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def print_detailed_metrics(result: Dict, strategy_name: str):
    """Print detailed metrics for a portfolio"""
    print(f"\n{strategy_name} Results:")
    print("-" * 50)
    
    # Basic metrics
    print(f"Expected Return: {result['return']:.4%}")
    print(f"Portfolio Risk: {result['risk']:.4%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    print(f"Turnover: {result['turnover']:.4%}")
    
    # Robust metrics
    if 'worst_case_return' in result:
        print(f"Worst-Case Return: {result['worst_case_return']:.4%}")
        print(f"Diversification Ratio: {result['diversification_ratio']:.4f}")
        print(f"Effective N: {result['effective_n']:.2f}")
    
    # Weight statistics
    weights = result['weights']
    print(f"\nWeight Statistics:")
    print(f"Max Weight: {weights.max():.4f}")
    print(f"Min Weight: {weights.min():.4f}")
    print(f"Weight Std: {weights.std():.4f}")
    
def plot_comprehensive_analysis(results: Dict, asset_names: pd.Index, stocks: Dict):
    """Create comprehensive visualization of results"""
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2)
    
    # 1. Risk-Return Plot
    ax1 = fig.add_subplot(gs[0, 0])
    risk_return_scatter(results, ax1)
    
    # 2. Weight Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    weight_distribution_plot(results, asset_names, ax2)
    
    # 3. Sector Allocations
    ax3 = fig.add_subplot(gs[1, :])
    sector_allocation_plot(results, stocks, ax3)
    
    # 4. Robust Metrics Comparison
    ax4 = fig.add_subplot(gs[2, :])
    robust_metrics_plot(results, ax4)
    
    plt.tight_layout()
    plt.show()

def run_portfolio_optimization_examples():
    """Comprehensive portfolio optimization examples with all strategies"""
    
    # 1. Get test data
    returns, stocks, benchmark_returns = get_test_data()
    n_assets = len(returns.columns)
    
    # 2. Initialize robust optimizer
    optimizer = RobustPortfolioOptimizer(
        returns=returns,
        uncertainty=0.1,           # Uncertainty parameter for robust optimization
        risk_aversion=1.0,        # Risk aversion parameter
        half_life=36,             # Half-life for exponential weighting
        risk_free_rate=0.02,      # Risk-free rate
        transaction_cost=0.001     # Transaction cost rate
    )
    
    # 3. Create different portfolio strategies
    portfolio_strategies = create_portfolio_strategies(n_assets, returns)
    
    # 4. Run optimizations and collect results
    all_results = {}
    for strategy_name, (objective, constraints, kwargs) in portfolio_strategies.items():
        print(f"\nTesting {strategy_name}")
        try:
            result = optimizer.optimize(
                objective=objective,
                constraints=constraints,
                **kwargs
            )
            
            # Add robust metrics
            robust_metrics = optimizer.calculate_robust_metrics(result['weights'])
            result.update(robust_metrics)
            
            all_results[strategy_name] = result
            print_detailed_metrics(result, strategy_name)
            
        except Exception as e:
            print(f"Strategy failed: {e}")
            all_results[strategy_name] = None
    
    # 5. Visualize results
    plot_comprehensive_analysis(all_results, returns.columns, stocks)
    
    return all_results

if __name__ == "__main__":
    result = run_portfolio_optimization_examples()