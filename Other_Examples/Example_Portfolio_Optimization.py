import pandas as pd
import numpy as np
from PortOpt import *
import seaborn as sns
from typing import Dict

################################################################################################################
#### Testing Portfolio Optimization
################################################################################################################

# Loading test data 
def get_multi_asset_test_data():
    """Get multi-asset universe data for testing"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Define multi-asset universe
    assets = {
        'Developed Equities': {
            'US Large Cap': ['SPY', 'QQQ', 'IWB'],  # S&P 500, Nasdaq, Russell 1000
            'US Small Cap': ['IWM', 'IJR'],         # Russell 2000, S&P 600
            'International': ['EFA', 'VEA', 'IEFA']  # EAFE Markets
        },
        'Emerging Markets': {
            'Broad EM': ['EEM', 'VWO'],  # Emerging Markets
            'Regional': ['FXI', 'EWZ']    # China, Brazil
        },
        'Fixed Income': {
            'Government': ['IEF', 'TLT', 'GOVT'],  # 7-10Y, 20Y+, All Treasury
            'Corporate': ['LQD', 'VCIT'],          # Investment Grade, Intermediate Corp
            'High Yield': ['HYG', 'JNK']           # High Yield Corporate
        },
        'Alternative': {
            'Real Estate': ['VNQ', 'REM'],     # REITs
            'Commodities': ['GLD', 'USO'],     # Gold, Oil
            'Alternatives': ['ABRYX', 'QAI']   # Absolute Return, Multi-Strategy
        }
    }
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6*365)  # 6 years of data
    
    # Download data
    prices = pd.DataFrame()
    asset_mapping = {}  # To store asset class mapping
    failed_downloads = []
    
    print("Downloading multi-asset data...")
    for asset_class, sub_classes in assets.items():
        for sub_class, symbols in sub_classes.items():
            for symbol in symbols:
                try:
                    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if len(data) > 0:
                        prices[symbol] = data['Adj Close']
                        asset_mapping[symbol] = {'class': asset_class, 'sub_class': sub_class}
                    else:
                        failed_downloads.append(symbol)
                except Exception as e:
                    print(f"Failed to download {symbol}: {e}")
                    failed_downloads.append(symbol)
    
    if failed_downloads:
        print(f"Warning: Failed to download data for: {', '.join(failed_downloads)}")
    
    # Resample to monthly frequency
    prices = prices.resample('M').last()
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Handle missing data and outliers
    returns = returns.fillna(method='ffill').fillna(method='bfill')
    
    # Remove extreme outliers using Winsorization
    for col in returns.columns:
        percentiles = returns[col].quantile([0.01, 0.99])
        returns[col] = returns[col].clip(percentiles[0.01], percentiles[0.99])
    
    # Get risk-free rate data
    try:
        tickers = yf.Tickers('^IRX')  # 13-week Treasury Bill rate
        rf_data = tickers.history(start=start_date, end=end_date)['Close'] / 100 / 12  # Monthly rate
        rf_rate = rf_data.mean()
    except:
        print("Warning: Using default risk-free rate")
        rf_rate = 0.02 / 12  # Default monthly risk-free rate
    
    # Calculate basic statistics
    print("\nAsset Universe Summary:")
    print(f"Number of assets: {len(returns.columns)}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    print(f"Number of periods: {len(returns)}")
    
    # Print asset class composition
    print("\nAsset Class Composition:")
    for asset_class in assets.keys():
        count = sum(1 for v in asset_mapping.values() if v['class'] == asset_class)
        print(f"{asset_class}: {count} assets")
    
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    # Print correlation summary
    print("\nCorrelation Summary:")
    print(f"Average correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
    
    return {
        'returns': returns,
        'asset_mapping': asset_mapping,
        'correlation': corr_matrix,
        'risk_free_rate': rf_rate,
        'prices': prices
    }

def analyze_universe(data: Dict):
    """Analyze the multi-asset universe"""
    returns = data['returns']
    asset_mapping = data['asset_mapping']
    
    # Calculate asset class level statistics
    stats = pd.DataFrame(columns=['Annual Return', 'Annual Vol', 'Sharpe Ratio', 'Max Drawdown'])
    
    for symbol in returns.columns:
        asset_info = asset_mapping[symbol]
        r = returns[symbol]
        
        # Calculate statistics
        annual_return = (1 + r.mean()) ** 12 - 1
        annual_vol = r.std() * np.sqrt(12)
        sharpe = (annual_return - data['risk_free_rate'] * 12) / annual_vol
        
        # Calculate drawdown
        cum_returns = (1 + r).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        stats.loc[symbol] = [annual_return, annual_vol, sharpe, max_drawdown]
    
    # Add asset class information
    stats['Asset Class'] = [asset_mapping[symbol]['class'] for symbol in stats.index]
    stats['Sub Class'] = [asset_mapping[symbol]['sub_class'] for symbol in stats.index]
    
    # Print summary statistics by asset class
    print("\nAsset Class Summary Statistics:")
    print(stats.groupby('Asset Class').mean().round(3))
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(2, 2)
    
    # 1. Risk-Return Scatter by Asset Class
    ax1 = fig.add_subplot(gs[0, 0])
    for asset_class in stats['Asset Class'].unique():
        mask = stats['Asset Class'] == asset_class
        ax1.scatter(
            stats.loc[mask, 'Annual Vol'],
            stats.loc[mask, 'Annual Return'],
            label=asset_class,
            alpha=0.7,
            s=100
        )
    ax1.set_xlabel('Annual Volatility')
    ax1.set_ylabel('Annual Return')
    ax1.set_title('Risk-Return by Asset Class')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Correlation Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(data['correlation'], cmap='RdYlBu_r', center=0, ax=ax2)
    ax2.set_title('Correlation Matrix')
    
    # 3. Drawdown Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    (-stats.groupby('Asset Class')['Max Drawdown'].mean()).plot(
        kind='bar',
        ax=ax3,
        color='darkred'
    )
    ax3.set_title('Average Maximum Drawdown by Asset Class')
    ax3.set_ylabel('Maximum Drawdown')
    plt.xticks(rotation=45)
    
    # 4. Sharpe Ratio Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    stats.groupby('Asset Class')['Sharpe Ratio'].mean().plot(
        kind='bar',
        ax=ax4,
        color='darkgreen'
    )
    ax4.set_title('Average Sharpe Ratio by Asset Class')
    ax4.set_ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return stats

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
        ),
        
        "9. Garlappi Robust": (
            ObjectiveFunction.GARLAPPI_ROBUST,
            OptimizationConstraints(
                long_only=True
            ),
            {'epsilon': 1.0, 'alpha': 1.0}
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
    res = get_multi_asset_test_data()
    returns = res['returns']
    n_assets = len(returns.columns)
    
    # 2. Initialize robust optimizer
    optimizer = RobustPortfolioOptimizer(
        returns=returns,
        epsilon=0.1,           # Uncertainty parameter for robust optimization
        alpha=1.0,        # Risk aversion parameter
        half_life=36,             # Half-life for exponential weighting
        risk_free_rate=0.0,      # Risk-free rate
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
    #plot_comprehensive_analysis(all_results, returns.columns, stocks)
    
    return all_results

if __name__ == "__main__":
    data = get_multi_asset_test_data()
    stats = analyze_universe(data)
    result = run_portfolio_optimization_examples()
    
    
