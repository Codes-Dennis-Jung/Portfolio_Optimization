import pandas as pd
import numpy as np
from PortOpt import *
from typing import Dict, List, Optional, Tuple, Union, Callable

################################################################################################################
#### Testing Efficient Frontier
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
    start_date = end_date - timedelta(days=15*365)  # 6 years of data
    
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

#### Efficient Frontier ####
def run_efficient_frontier_analysis(returns: pd.DataFrame, asset_mapping: Dict):
    """Run comprehensive efficient frontier analysis with asset-specific risk aversion"""
    print("\nInitializing Robust Efficient Frontier Calculator...")
    
    # Define asset-specific alphas based on asset classes
    alpha_by_class = {
        'Developed Equities': 1.0,    # Base risk aversion for developed markets
        'Emerging Markets': 1.5,      # Higher risk aversion for emerging markets
        'Fixed Income': 0.5,          # Lower risk aversion for fixed income
        'Alternative': 1.2            # Moderately higher risk aversion for alternatives
    }
    
    # Create alpha vector based on asset mapping
    alphas = np.array([
        alpha_by_class[asset_mapping[col]['class']] 
        for col in returns.columns
    ])
    
    # Print alpha assignments
    print("\nAsset-specific risk aversion (alpha) assignments:")
    for col in returns.columns:
        asset_class = asset_mapping[col]['class']
        print(f"{col} ({asset_class}): {alpha_by_class[asset_class]:.2f}")
    
    # Initialize frontier calculator with different uncertainty levels
    base_epsilons = [0]
    frontier_results = {}
    
    for base_epsilon in base_epsilons:
        print(f"\nComputing Efficient Frontier with base epsilon = {base_epsilon:.2f}")
        
        try:
            # Initialize calculator with asset-specific alphas
            calculator = RobustEfficientFrontier(
                returns=returns,
                epsilon=base_epsilon,     # Base uncertainty parameter
                alpha=alphas,             # Asset-specific risk aversion vector
                half_life=36,             # Half-life for exponential weighting
                risk_free_rate=0.0,       # Risk-free rate
                transaction_cost=0.001    # Transaction cost rate
            )
            
            # Define constraints with asset class grouping
            base_constraints = OptimizationConstraints(
                long_only=True,
                box_constraints={i: (0.0, 0.3) for i in range(len(returns.columns))},
                group_constraints={
                    asset_class: GroupConstraint(
                        assets=[i for i, col in enumerate(returns.columns)
                               if asset_mapping[col]['class'] == asset_class],
                        bounds=(0.1, 0.4)  # 10-40% per asset class
                    )
                    for asset_class in alpha_by_class.keys()
                }
            )
            
            # Compute frontier
            epsilon_range = (base_epsilon * 0.5, base_epsilon * 1.5)
            alpha_scale_range = (0.5, 1.5)  # Range to scale alpha vector
            
            results = calculator.compute_efficient_frontier(
                n_points=30,
                constraints=base_constraints,
                epsilon_range=epsilon_range,
                alpha_scale_range=alpha_scale_range,
                risk_range=None
            )
            
            frontier_results[base_epsilon] = results
            
            # Plot individual frontier
            calculator.plot_frontier(results)
            
            # Print asset class allocations
            print_asset_class_allocations(results, returns.columns, asset_mapping)
            
        except Exception as e:
            print(f"Error computing frontier for base epsilon {base_epsilon}: {str(e)}")
            continue
    
    return frontier_results

def print_asset_class_allocations(results: Dict[str, np.ndarray], 
                                columns: pd.Index, 
                                asset_mapping: Dict):
    """Print asset class allocations for key portfolios"""
    print("\nAsset Class Allocations:")
    print("-" * 50)
    
    # Get key portfolio indices
    max_sharpe_idx = np.argmax(results['robust_sharpe_ratios'])
    min_risk_idx = np.argmin(results['risks'])
    max_return_idx = np.argmax(results['returns'])
    
    portfolios = {
        'Maximum Robust Sharpe': results['weights'][max_sharpe_idx],
        'Minimum Risk': results['weights'][min_risk_idx],
        'Maximum Return': results['weights'][max_return_idx]
    }
    
    for name, weights in portfolios.items():
        print(f"\n{name} Portfolio:")
        print("-" * 30)
        
        # Calculate allocations by asset class
        class_alloc = {}
        for i, col in enumerate(columns):
            asset_class = asset_mapping[col]['class']
            class_alloc[asset_class] = class_alloc.get(asset_class, 0) + weights[i]
        
        # Print allocations
        for asset_class, alloc in class_alloc.items():
            print(f"{asset_class}: {alloc:.2%}")

def run_comprehensive_tests():
    """Run both general portfolio optimization and efficient frontier tests"""
    # Get test data
    res = get_multi_asset_test_data()
    returns = res['returns']
    asset_mapping = res['asset_mapping']
    
    print("\nRunning Efficient Frontier Analysis...")
    frontier_results = run_efficient_frontier_analysis(returns, asset_mapping)
    
    return frontier_results, asset_mapping

def print_frontier_metrics(frontier_results: Dict[float, Dict[str, np.ndarray]], 
                         asset_mapping: Dict):
    """Print summary metrics for each frontier with asset class breakdowns"""
    print("\nGarlappi Efficient Frontier Summary Metrics:")
    print("-" * 50)
    
    for base_epsilon, results in frontier_results.items():
        print(f"\nBase Epsilon Level: {base_epsilon:.2f}")
        print("-" * 30)
        
        # Calculate key metrics
        max_robust_sharpe_idx = np.argmax(results['robust_sharpe_ratios'])
        min_risk_idx = np.argmin(results['risks'])
        max_return_idx = np.argmax(results['returns'])
        
        # Function to print portfolio details
        def print_portfolio_details(idx, name):
            print(f"\n{name} Portfolio:")
            print(f"Return: {results['returns'][idx]:.4%}")
            print(f"Risk: {results['risks'][idx]:.4%}")
            print(f"Robust Sharpe: {results['robust_sharpe_ratios'][idx]:.4f}")
            print(f"Worst-Case Return: {results['worst_case_returns'][idx]:.4%}")
            print(f"Estimation Uncertainty: {results['estimation_uncertainty'][idx]:.4%}")
            
            # Print asset class allocations
            weights = results['weights'][idx]
            class_alloc = {}
            for i, weight in enumerate(weights):
                asset_class = asset_mapping[list(asset_mapping.keys())[i]]['class']
                class_alloc[asset_class] = class_alloc.get(asset_class, 0) + weight
            
            print("\nAsset Class Allocations:")
            for asset_class, alloc in class_alloc.items():
                print(f"{asset_class}: {alloc:.2%}")
        
        # Print details for each portfolio type
        print_portfolio_details(max_robust_sharpe_idx, "Maximum Robust Sharpe Ratio")
        print_portfolio_details(min_risk_idx, "Minimum Risk")
        print_portfolio_details(max_return_idx, "Maximum Return")

if __name__ == "__main__":
    # Run complete analysis
    frontier_results, asset_mapping = run_comprehensive_tests()