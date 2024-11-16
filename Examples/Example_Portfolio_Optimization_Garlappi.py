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

def plot_comprehensive_analysis(results: Dict, asset_names: pd.Index, asset_mapping: Dict):
    """Create comprehensive visualization of Garlappi optimization results with error handling"""
    # Check if we have any valid results
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        print("No valid optimization results to plot.")
        return
    
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2)
    
    try:
        # 1. Risk-Return Plot with Robust Sharpe Ratios
        ax1 = fig.add_subplot(gs[0, 0])
        plot_risk_return_robust(valid_results, ax1)
        
        # 2. Worst-Case Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        plot_worst_case_analysis(valid_results, ax2)
        
        # 3. Asset Class Allocations
        ax3 = fig.add_subplot(gs[1, :])
        plot_asset_class_allocations(valid_results, asset_names, asset_mapping, ax3)
        
        # 4. Uncertainty and Diversification
        ax4 = fig.add_subplot(gs[2, 0])
        plot_uncertainty_metrics(valid_results, ax4)
        
        # 5. Risk Contributions
        ax5 = fig.add_subplot(gs[2, 1])
        plot_risk_contributions(valid_results, asset_names, asset_mapping, ax5)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        plt.close(fig)

def plot_risk_return_robust(results: Dict, ax: plt.Axes):
    """Plot risk-return scatter with robust Sharpe ratio bubbles and improved axis scaling"""
    if not results:
        ax.text(0.5, 0.5, 'No valid results to plot', 
                ha='center', va='center', transform=ax.transAxes)
        return
        
    risks = [r['risk'] for r in results.values()]
    returns = [r['return'] for r in results.values()]
    robust_sharpes = [r.get('robust_sharpe', r.get('sharpe_ratio', 0)) for r in results.values()]
    
    # Handle single point case
    if len(risks) == 1:
        sc = ax.scatter(risks, returns, s=500, c='blue', alpha=0.6)
    else:
        # Normalize bubble sizes
        size_scale = 1000
        sizes = np.array(robust_sharpes)
        size_range = sizes.max() - sizes.min()
        if size_range > 0:
            sizes = (sizes - sizes.min()) / size_range * size_scale + 100
        else:
            sizes = [500] * len(sizes)  # Default size if all values are the same
        
        sc = ax.scatter(risks, returns, s=sizes, c=robust_sharpes, 
                       cmap='viridis', alpha=0.6)
        plt.colorbar(sc, ax=ax, label='Robust Sharpe Ratio')
    
    # Add annotations
    for i, (name, _) in enumerate(results.items()):
        ax.annotate(name, (risks[i], returns[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    # Calculate axis limits with padding
    risk_range = max(risks) - min(risks)
    return_range = max(returns) - min(returns)
    
    risk_padding = risk_range * 0.2  # 20% padding
    return_padding = return_range * 0.2  # 20% padding
    
    # Set axis limits
    ax.set_xlim([max(0, min(risks) - risk_padding), 
                 max(risks) + risk_padding])
    ax.set_ylim([min(returns) - return_padding,
                 max(returns) + return_padding])
    
    # Format axis labels with percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    # Add labels and grid
    ax.set_xlabel('Risk (Volatility)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.set_title('Risk-Return Trade-off with Robust Sharpe Ratios', fontsize=14, pad=20)
    
    # Enhance grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add minor gridlines
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    ax.minorticks_on()
    
    # Make spines visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    return ax

def plot_worst_case_analysis(results: Dict, ax: plt.Axes):
    """Plot comparison of expected vs worst-case returns"""
    if not results:
        ax.text(0.5, 0.5, 'No valid results to plot', 
                ha='center', va='center', transform=ax.transAxes)
        return
        
    strategies = list(results.keys())
    expected_returns = [r['return'] for r in results.values()]
    worst_case_returns = [r.get('worst_case_return', r['return']) for r in results.values()]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax.bar(x - width/2, expected_returns, width, label='Expected Return',
           color='blue', alpha=0.6)
    ax.bar(x + width/2, worst_case_returns, width, label='Worst-Case Return',
           color='red', alpha=0.6)
    
    ax.set_ylabel('Return')
    ax.set_title('Expected vs Worst-Case Returns by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.grid(True)

def plot_asset_class_allocations(results: Dict, asset_names: pd.Index, 
                               asset_mapping: Dict, ax: plt.Axes):
    """Plot asset class allocations for each strategy"""
    if not results:
        ax.text(0.5, 0.5, 'No valid results to plot', 
                ha='center', va='center', transform=ax.transAxes)
        return
        
    # Calculate asset class allocations
    class_allocations = {}
    for strategy, result in results.items():
        weights = result['weights']
        class_alloc = {}
        for i, asset in enumerate(asset_names):
            asset_class = asset_mapping[asset]['class']
            class_alloc[asset_class] = class_alloc.get(asset_class, 0) + weights[i]
        class_allocations[strategy] = class_alloc
    
    df = pd.DataFrame(class_allocations).T
    if not df.empty:
        df.plot(kind='bar', stacked=True, ax=ax)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Allocation')
        ax.set_title('Asset Class Allocations by Strategy')
        ax.legend(title='Asset Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def plot_uncertainty_metrics(results: Dict, ax: plt.Axes):
    """Plot uncertainty and diversification metrics"""
    if not results:
        ax.text(0.5, 0.5, 'No valid results to plot', 
                ha='center', va='center', transform=ax.transAxes)
        return
        
    strategies = list(results.keys())
    uncertainty = [r.get('estimation_uncertainty', 0) for r in results.values()]
    div_ratio = [r.get('diversification_ratio', 0) for r in results.values()]
    
    ax2 = ax.twinx()
    
    l1 = ax.plot(strategies, uncertainty, 'b-o', label='Estimation Uncertainty')
    l2 = ax2.plot(strategies, div_ratio, 'r-s', label='Diversification Ratio')
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Estimation Uncertainty', color='blue')
    ax2.set_ylabel('Diversification Ratio', color='red')
    
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)
    
    ax.set_title('Uncertainty and Diversification Metrics')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def plot_risk_contributions(results: Dict, asset_names: pd.Index, 
                          asset_mapping: Dict, ax: plt.Axes):
    """Plot risk contributions by asset class"""
    if not results:
        ax.text(0.5, 0.5, 'No valid results to plot', 
                ha='center', va='center', transform=ax.transAxes)
        return
        
    # Calculate risk contributions by asset class
    class_contributions = {}
    for strategy, result in results.items():
        risk_contrib = result.get('risk_contributions', result['weights'] * result['risk'])
        class_contrib = {}
        for i, asset in enumerate(asset_names):
            asset_class = asset_mapping[asset]['class']
            class_contrib[asset_class] = class_contrib.get(asset_class, 0) + risk_contrib[i]
        class_contributions[strategy] = class_contrib
    
    df = pd.DataFrame(class_contributions).T
    if not df.empty:
        df.plot(kind='bar', stacked=True, ax=ax)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Risk Contribution')
        ax.set_title('Risk Contributions by Asset Class')
        ax.legend(title='Asset Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def create_portfolio_strategies(n_assets: int, returns: pd.DataFrame, asset_mapping: Dict) -> Dict:
    """Create different portfolio strategies with asset-specific alphas"""
    
    # Define asset-specific alphas based on asset classes
    alpha_by_class = {
        'Developed Equities': 0.9,    # Base risk aversion
        'Emerging Markets': 1.5,      # Higher risk aversion
        'Fixed Income': 0.5,          # Lower risk aversion
        'Alternative': 1.2            # Moderate risk aversion
    }
    
    # Create alpha vector
    alphas = np.array([
        alpha_by_class[asset_mapping[col]['class']] 
        for col in returns.columns
    ])
    
    # Create group constraints by asset class
    group_constraints = {
        asset_class: GroupConstraint(
            assets=[i for i, col in enumerate(returns.columns)
                   if asset_mapping[col]['class'] == asset_class],
            bounds=(0.1, 0.4)  # 10-40% per asset class
        )
        for asset_class in alpha_by_class.keys()
    }
    
    # Generate stress scenarios
    scenarios = generate_scenarios(returns)
    
    strategies = {
        "1. Garlappi Robust (Asset-Specific)": (
            ObjectiveFunction.GARLAPPI_ROBUST,
            OptimizationConstraints(
                long_only=True,
                group_constraints=group_constraints,
                box_constraints={i: (0.0, 0.25) for i in range(n_assets)}
            ),
            {
                'epsilon': 0.1,
                'alpha': alphas,
                'omega_method': 'bayes'
            }
        ),
        
        "2. Garlappi Conservative": (
            ObjectiveFunction.GARLAPPI_ROBUST,
            OptimizationConstraints(
                long_only=True,
                group_constraints=group_constraints,
                box_constraints={i: (0.0, 0.25) for i in range(n_assets)}
            ),
            {
                'epsilon': 0.2,
                'alpha': alphas * 1.5,  # More conservative
                'omega_method': 'factor'
            }
        ),
        
        "3. Garlappi Aggressive": (
            ObjectiveFunction.GARLAPPI_ROBUST,
            OptimizationConstraints(
                long_only=True,
                group_constraints=group_constraints,
                box_constraints={i: (0.0, 0.25) for i in range(n_assets)}
            ),
            {
                'epsilon': 0.05,
                'alpha': alphas * 0.7,  # More aggressive
                'omega_method': 'bayes'
            }
        ),
        
        "4. Risk-Based Robust": (
            ObjectiveFunction.GARLAPPI_ROBUST,
            OptimizationConstraints(
                long_only=True,
                target_risk=0.08,
                group_constraints=group_constraints,
                box_constraints={i: (0.0, 0.25) for i in range(n_assets)}
            ),
            {
                'epsilon': 0.1,
                'alpha': alphas,
                'omega_method': 'factor'
            }
        ),
        
        "5. Return-Based Robust": (
            ObjectiveFunction.GARLAPPI_ROBUST,
            OptimizationConstraints(
                long_only=True,
                target_return=0.08,
                group_constraints=group_constraints,
                box_constraints={i: (0.0, 0.25) for i in range(n_assets)}
            ),
            {
                'epsilon': 0.1,
                'alpha': alphas,
                'omega_method': 'bayes'
            }
        )
    }
    
    return strategies

def run_portfolio_optimization_examples():
    """Comprehensive portfolio optimization examples with Garlappi framework"""
    
    # 1. Get test data
    res = get_multi_asset_test_data()
    returns = res['returns']
    asset_mapping = res['asset_mapping']
    n_assets = len(returns.columns)
    
    # Print asset class composition and alphas
    print("\nAsset-Specific Risk Aversion (Alpha) Assignments:")
    alpha_by_class = {
        'Developed Equities': 0.8,
        'Emerging Markets': 1.5,
        'Fixed Income': 0.5,
        'Alternative': 1.2
    }
    for col in returns.columns:
        asset_class = asset_mapping[col]['class']
        print(f"{col} ({asset_class}): {alpha_by_class[asset_class]:.2f}")
    
    # 2. Initialize robust optimizer
    optimizer = RobustEfficientFrontier(
        returns=returns,
        epsilon=0.1,
        alpha=np.array([alpha_by_class[asset_mapping[col]['class']] 
                       for col in returns.columns]),
        half_life=36,
        risk_free_rate=0.0,
        transaction_cost=0.001
    )

    # 3. Create different portfolio strategies
    portfolio_strategies = create_portfolio_strategies(n_assets, returns, asset_mapping)
    
    # 4. Run optimizations and collect results
    all_results = {}
    for strategy_name, (objective, constraints, kwargs) in portfolio_strategies.items():
        print(f"\nTesting {strategy_name}")
        try:
            # Run optimization
            result = optimizer.optimize(
                objective=objective,
                constraints=constraints,
                **kwargs
            )
            
            # Calculate robust metrics
            robust_metrics = optimizer.calculate_robust_metrics(
                result['weights'],
                kwargs.get('alpha', optimizer.alpha)
            )
            
            # Update result with robust metrics
            result.update(robust_metrics)
            
            # Calculate robust Sharpe ratio if not present
            if 'robust_sharpe' not in result:
                result['robust_sharpe'] = ((result['worst_case_return'] - optimizer.risk_free_rate) / 
                                         result['risk'])
            
            # Print asset class allocations
            print("\nAsset Class Allocations:")
            class_alloc = {}
            for i, weight in enumerate(result['weights']):
                asset_class = asset_mapping[returns.columns[i]]['class']
                class_alloc[asset_class] = class_alloc.get(asset_class, 0) + weight
            
            for asset_class, alloc in class_alloc.items():
                print(f"{asset_class}: {alloc:.2%}")
            
            all_results[strategy_name] = result
            print_detailed_metrics(result, strategy_name)
            
        except Exception as e:
            print(f"Strategy failed: {str(e)}")
            continue
    
    if not all_results:
        print("No strategies succeeded. Cannot generate plots.")
        return None, asset_mapping
    
    # 5. Create plots
    print("\nGenerating visualization...")
    plot_comprehensive_analysis(all_results, returns.columns, asset_mapping)
    
    return all_results, asset_mapping

def print_detailed_metrics(result: Dict, strategy_name: str):
    """Print detailed metrics for a portfolio with robust measures"""
    print(f"\n{strategy_name} Results:")
    print("-" * 50)
    
    # Basic metrics
    print(f"Expected Return: {result['return']:.4%}")
    print(f"Portfolio Risk: {result['risk']:.4%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    print(f"Turnover: {result.get('turnover', 0):.4%}")
    
    # Robust metrics (with fallbacks)
    print(f"Robust Sharpe Ratio: {result.get('robust_sharpe', result['sharpe_ratio']):.4f}")
    print(f"Worst-Case Return: {result.get('worst_case_return', result['return']):.4%}")
    print(f"Estimation Uncertainty: {result.get('estimation_uncertainty', 0):.4%}")
    print(f"Diversification Ratio: {result.get('diversification_ratio', 1):.4f}")
    print(f"Effective N: {result.get('effective_n', 1):.2f}")
    
    # Weight statistics
    weights = result['weights']
    print(f"\nWeight Statistics:")
    print(f"Max Weight: {weights.max():.4f}")
    print(f"Min Weight: {weights.min():.4f}")
    print(f"Weight Std: {weights.std():.4f}")

def plot_risk_return_robust(results: Dict, ax: plt.Axes):
    """Plot risk-return scatter with robust Sharpe ratio bubbles"""
    if not results:
        ax.text(0.5, 0.5, 'No valid results to plot', 
                ha='center', va='center', transform=ax.transAxes)
        return
        
    risks = [r['risk'] for r in results.values()]
    returns = [r['return'] for r in results.values()]
    robust_sharpes = [r.get('robust_sharpe', r['sharpe_ratio']) for r in results.values()]
    
    # Handle single point case
    if len(risks) == 1:
        sc = ax.scatter(risks, returns, s=500, c='blue', alpha=0.6)
    else:
        # Normalize bubble sizes
        size_scale = 1000
        sizes = np.array(robust_sharpes)
        size_range = sizes.max() - sizes.min()
        if size_range > 0:
            sizes = (sizes - sizes.min()) / size_range * size_scale + 100
        else:
            sizes = [500] * len(sizes)  # Default size if all values are the same
        
        sc = ax.scatter(risks, returns, s=sizes, c=robust_sharpes, 
                       cmap='viridis', alpha=0.6)
        plt.colorbar(sc, ax=ax, label='Robust Sharpe Ratio')
    
    for i, (name, _) in enumerate(results.items()):
        ax.annotate(name, (risks[i], returns[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Risk (Volatility)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Risk-Return Trade-off with Robust Sharpe Ratios')
    ax.grid(True)

if __name__ == "__main__":
    results, asset_mapping = run_portfolio_optimization_examples()